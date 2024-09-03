#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AffineExpr.h"

#define DEBUG_TYPE "raise-to-linalg"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace affine;

namespace {
struct RaiseAffineToLinalg : public AffineRaiseToLinalgBase<RaiseAffineToLinalg> {
  void runOnOperation() override;
};
} // namespace

// Also want to add support for affine.for ( ) { linalg.generic } -> bigger linalg.generic
// Also probably want to try to do { linalg.generc1(); linalg.generic2(); } -> bigger linalg.generic()

/*

affine.for() {
    affine.for() {
    } 
    affine.for() {
    }
}

*/
struct Condition {
    bool ifTrue;
    AffineIfOp op;
    Condition(bool ifTrue, AffineIfOp op) : ifTrue(ifTrue), op(op) {}
};

bool isLinearInIndex(AffineExpr expr, size_t idx) {
    if (!expr.isFunctionOfDim(idx)) {
        return true;
    }

    if (expr.getKind() == AffineExprKind::DimId) {
        return true;
    }

        if (expr.getKind() == AffineExprKind::Add) {
            auto binop = expr.cast<AffineBinaryOpExpr>();
            return isLinearInIndex(binop.getLHS(), idx) && isLinearInIndex(binop.getRHS(), idx);
        }
        if (expr.getKind() == AffineExprKind::Mul) {
            auto binop = expr.cast<AffineBinaryOpExpr>();
            return (isLinearInIndex(binop.getLHS(), idx) && !binop.getRHS().isFunctionOfDim(idx)) ||
                    (isLinearInIndex(binop.getRHS(), idx) && !binop.getLHS().isFunctionOfDim(idx));
        }

    return false;
}

bool isLinearInIndex(AffineMap map, size_t idx) {
    for (auto expr : map.getResults()) {
        if (!isLinearInIndex(expr, idx))
            return false;
    }
    return true;
}

 AffineExpr shiftDimsDown1(AffineExpr expr, unsigned numDims,
                                  unsigned offset) {
   SmallVector<AffineExpr, 4> dims;
   for (unsigned idx = 0; idx < offset; ++idx)
     dims.push_back(getAffineDimExpr(idx, expr.getContext()));
   for (unsigned idx = offset; idx < numDims; ++idx)
     dims.push_back(getAffineDimExpr(idx - 1, expr.getContext()));
   return expr.replaceDimsAndSymbols(dims, {});
 }

//This is reducing the number of input dims in expression by 1
 AffineMap shiftDimsDown1(AffineMap expr, unsigned numDim,
                                  unsigned offset) {
            assert(offset <= expr.getNumDims());
     return AffineMap::get(expr.getNumDims() - 1, expr.getNumSymbols(),
                           llvm::map_to_vector<4>(
                               expr.getResults(),
                               [&](AffineExpr e) {
                                 return shiftDimsDown1(e, expr.getNumDims(), offset);
                               }),
                           expr.getContext());
                                  }

// Given an affine map `oldmap`, memref `val`, and corresponding input values (which are a list of indicies, then symbols),
// and a loop index `ind` produce the following:
//  1. A (potentially new) memref value `newval` which does not have any dependence on `ind`
//     and
//  2. an affine map `newmap` which takes a single index (`ind`) and produces indices into `newval` such that
//     indexing `newval[map(ind)]` produces the same result as indexing the original map.
std::pair<Value, AffineMap> remap_in_affine_dim(bool &legal, OpBuilder &builder, AffineMap oldmap, Value val, Value idx, Value idx_size, int loopLowerBound, int loopStepSize, mlir::OperandRange vals) {
    // First we need to remove any dependence on the loop index from the affine map
    SmallVector<Value> vals_without_idx;
    ssize_t dim_idx = -1;
    //To check if induction variable of for loop in an operand of this op (load/store)
    for (auto &&[i, v] : llvm::enumerate(vals)) {
        if (v == idx) {
            // Offset we're replacing must be an index (not a symbol).
            // If we guarantee to run AffineCFG first, this should always be true.
            assert(i < oldmap.getNumDims());
            // There should only be one use of the index.
            assert(dim_idx == -1);
            dim_idx = i;
            continue;
        }
        vals_without_idx.push_back(v);
    }

    if (dim_idx != -1 && !isLinearInIndex(oldmap, dim_idx)) {
        legal = false;
        return {val, oldmap};
    }


    // Evaluate offsets as oldmap replacing idx with 0, and evaluating at the remaining variables

    //Instead of lower bound we are using 0 (assumption as the lower bound)
    AffineMap offsetMap = oldmap;
    if (dim_idx != -1) {
        offsetMap = oldmap.replace(builder.getAffineDimExpr(dim_idx), builder.getAffineConstantExpr(loopLowerBound),offsetMap.getNumDims(), offsetMap.getNumSymbols());
        offsetMap = shiftDimsDown1(offsetMap, oldmap.getNumDims(), dim_idx);
    }

    //Instead of using loop step we are using 1 (Assumption as the stride size)
    AffineMap strideMap = oldmap;
    if (dim_idx != -1) {
        strideMap = oldmap.replace(builder.getAffineDimExpr(dim_idx), builder.getAffineConstantExpr(loopLowerBound + loopStepSize),strideMap.getNumDims(), strideMap.getNumSymbols());
        strideMap = shiftDimsDown1(strideMap, oldmap.getNumDims(), dim_idx);
    }

    //Subtracting maps of stride and offset, gives you the offset value in the result of the map
    {
        SmallVector<AffineExpr> subtracts;
        for (auto &&[lhs, rhs] : llvm::zip(strideMap.getResults(), offsetMap.getResults())) {
            subtracts.push_back(lhs - rhs);
        }
        strideMap = AffineMap::get(offsetMap.getNumDims(), offsetMap.getNumSymbols(), subtracts, builder.getContext());
    }

    // Expression to index into the generated subview given the loop index
    SmallVector<AffineExpr> loop_idxs;

    // List of starting offsets into the subview
    SmallVector<Value> offsets;
    SmallVector<Value> sizes;
    SmallVector<Value> strides;

    for (auto &&[expr, offset_expr, stride_expr] : llvm::zip(oldmap.getResults(), offsetMap.getResults(),strideMap.getResults() )) {
        offsets.push_back(builder.create<affine::AffineApplyOp>(val.getLoc(),AffineMap::get(offsetMap.getNumDims(), offsetMap.getNumSymbols(), offset_expr, builder.getContext()), vals_without_idx)); //What is there are symbols in the expression?
        strides.push_back(builder.create<affine::AffineApplyOp>(val.getLoc(),AffineMap::get(strideMap.getNumDims(), strideMap.getNumSymbols(), stride_expr, builder.getContext()), vals_without_idx)); //What is there are symbols in the expression?
        if (!expr.isFunctionOfDim(dim_idx)) {
            loop_idxs.push_back(builder.getAffineConstantExpr(0));
            sizes.push_back(builder.create<arith::ConstantIndexOp>(val.getLoc(), 1));
        } else {
            loop_idxs.push_back(builder.getAffineDimExpr(0));
            sizes.push_back(idx_size);
        }
    }

    auto newval = builder.create<memref::SubViewOp>(val.getLoc(), val, offsets, sizes, strides);
    legal = true;
    //Does this need fix? Here we are constraining to dims as 1 and symbols as 0, should it be, original 
    return {newval, AffineMap::get(/*dims*/1, /*symbols*/0, loop_idxs, builder.getContext())};
}


// store A[...]
// val = load A[...]

/*  prevA : 
    store A
    val is now prevA
*/


struct AffineForOpRaising : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp loop,
                                PatternRewriter &rewriter) const final {

    // Don't handle accumulations in registers for the moment, we can have
    // a separate pattern move them into memref's
    if (loop.getNumResults() != 0) {
        return failure();
    }

    SmallVector<std::pair<std::vector<Condition>, AffineLoadOp>> loads;
    SmallVector<std::pair<std::vector<Condition>, AffineStoreOp>> stores;
    // TODO Also collect all the linalg generics!

    // Check that the only operations within the region are either:
    //      affine.load, affine.store, affine.if, affine.yield
    // Additionally, for each load/store, remember what conditions are
    // required for that load or store to execute.
    auto result = loop->walk<WalkOrder::PreOrder>([&](Operation* op) {
        if (op == loop) return WalkResult::advance();
        // TODO extend this, any non-memory operation is also legal here.
        // mul, add, etc (we can just check propety)
        if (isa<AffineYieldOp, AffineIfOp>(op)) {
            return WalkResult::advance();
        }
        if (isa<AffineLoadOp, AffineStoreOp>(op)) {
            Operation *cur = op->getParentOp();
            std::vector<Condition> conditions;
            while (cur != loop) {
                auto ifstmt = dyn_cast<AffineIfOp>(cur);
                if (!ifstmt) {
                    return WalkResult::interrupt();
                }
                bool ifTrue = ifstmt.getThenRegion().isAncestor(cur->getParentRegion());
                conditions.emplace_back(ifTrue, ifstmt);
                cur = ifstmt->getParentOp();
            }
            if (auto load = dyn_cast<AffineLoadOp>(op)) {
                loads.emplace_back(conditions, load);
            } else {
                auto store = cast<AffineStoreOp>(op);
                stores.emplace_back(conditions, store);
            }
            return WalkResult::advance();
        }
        if (isReadNone(op)) {
            return WalkResult::advance();
        }
        return WalkResult::interrupt();
    });
    
    if (result.wasInterrupted()) return failure();

    DominanceInfo DI(loop);

    // Check that all of the stores do not alias the loaded values (otherwise we could get an incorrect result)
    // TODO we can extend this and handle things like reductions, but we're going to start easy for now
    // TODO 
    DenseMap<AffineLoadOp, AffineStoreOp> stores_map;
    for (auto &&[_, store] : stores) {
        for (auto &&[_, load]: loads) {
            if (mayAlias(load.getMemref(), store.getMemref())) {
                // We have one exception in this case -- if the load and store are from the exact same location, it is permitted.
                if (load.getMemref() == store.getMemref() &&
                    load.getAffineMap() == store.getAffineMap() &&
                    load.getIndices() == store.getIndices() && DI.dominates((Operation*)load,(Operation*)store)) {
                        stores_map[load] = store;
                        continue;
                    }
                return failure();
            }
        }
        for (auto &&[_, store2]: stores) {
            if (store == store2) continue;
            if (mayAlias(store.getMemref(), store2.getMemref())) {
                return failure();
            }
        }
    }
    // Check that any other loads / stores do not alias with any linalg generics
    // We're going to need to upgrade the defn of mayAlias for subviews (aka mayAlias(subview, x) -> mayAlias(operand(subview), x))

    SmallVector<Value> inputs;
    SmallVector<AffineMap> affineMaps;

    //if (loop.getStep() != 1) {
    //    return failure();
    //}

    // our remapper currently assumes 0 start to bound. 
    if (!loop.hasConstantLowerBound() /*|| loop.getConstantLowerBound() != 0*/) {
        return failure();
    }

    // compute this correctly later.
    auto ubMap = loop.getUpperBoundMap();
    auto ubOperands = loop.getUpperBoundOperands();
    if (!ubMap || ubMap.getNumResults() != 1) return failure();

    // Retrieve the lower bound
    auto lbMap = loop.getLowerBoundMap();
    auto lbOperands = loop.getLowerBoundOperands();
    if (!lbMap || lbMap.getNumResults() != 1) return failure();
    
    auto ub = loop.getSingleUpperBound();
    if (!ub) return failure();

    auto lb = loop.getSingleLowerBound();
    if (!lb) return failure();
    

    if (!loop.hasConstantUpperBound()) {
        return failure();
    }

    // Retrieve the step size
    int64_t step = loop.getStep();

    // Get the single result expressions
    AffineExpr ubExpr = ubMap.getResult(0);
    auto ubValue = rewriter.create<AffineApplyOp>(loop.getLoc(), ubMap, ubOperands);
    
    AffineExpr lbExpr = lbMap.getResult(0);
    auto lbValue = rewriter.create<AffineApplyOp>(loop.getLoc(), lbMap, lbOperands);

    //// Ensure the bounds are constant expressions
    auto ubConst = ubExpr.dyn_cast<AffineConstantExpr>();
    auto lbConst = lbExpr.dyn_cast<AffineConstantExpr>();
    if (!ubConst || !lbConst) return failure();

    // Compute the loop size
    //int64_t loopSize = ubConst.getValue() - lbConst.getValue();
    auto loopSize = rewriter.create<SubIOp>(loop.getLoc(), ubValue, lbValue);
    
    //Value loopSize = rewriter.create<arith::ConstantIndexOp>(loop.getLoc(), loop.getConstantUpperBound());//rewriter.create<arith::SubIOp>(loop.getLoc(), *ub, *lb);
    
    // current spec is going to be indexed off of the loop var in isolation
    for (auto &&[conds, load] : loads) {
        // Only support unconditional loads for the moment
        if (conds.size() != 0) return failure();

        if (stores_map.find(load) != stores_map.end()) {
            // We have a store that represents this load.
            continue;
        }

        bool legal = true;
       
        auto &&[newMemref, newAffineMap] = remap_in_affine_dim(legal, rewriter, load.getAffineMap(), load.getMemref(), loop.getInductionVar(),
        loopSize, lbConst.getValue(), step, load.getMapOperands());

        if (!legal) return failure();

        affineMaps.push_back(newAffineMap);
        inputs.push_back(newMemref);
    }
    // TODO Push all of the inputs to the linalg generics (modifying maps as needed)
    
    SmallVector<Value> outputs;
    // Store we may need to reindex into a splat potentially later, but for now we'll be lazy
    for (auto &&[conds, store] : stores) {
        // Only support unconditional loads for the moment
        if (conds.size() != 0) return failure();

        bool legal = true;
       
        auto &&[newMemref, newAffineMap] = remap_in_affine_dim(legal, rewriter, store.getAffineMap(), store.getMemref(), loop.getInductionVar(),
        loopSize, lbConst.getValue(), step, store.getMapOperands());

        if (!legal) return failure();

        affineMaps.push_back(newAffineMap);
        outputs.push_back(newMemref);
    }
    // TODO Push all of the outputs to the linalg generics

    // TODO presently  if linalg generic exists, assert there are no load/stores
    // TODO assert only zero or one linalg generic exists
    SmallVector<utils::IteratorType> iteratorTypes;
    // TODO if linalg generic exists, make this iterator type prepend to the existing iterators
    iteratorTypes.push_back((stores_map.size() == 0) ? utils::IteratorType::parallel : utils::IteratorType::reduction);



    StringAttr empty = StringAttr::get(loop.getContext());
    auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
      loop.getLoc(), TypeRange(), inputs, outputs, affineMaps, iteratorTypes,
      empty,
      empty);

    // TODO if doing the linalg generic case, ignore a lot of the below and instead of injecting the old body of the affine.for, move the inner linalg.generic body 
    // and also add a new induction variable
    auto blk = &*loop.getRegion().begin();
    rewriter.setInsertionPointToStart(blk);

    // This index will replace the use of the affine index
    auto idx = rewriter.create<linalg::IndexOp>(loop.getLoc(), rewriter.getIndexAttr(0));
    rewriter.replaceAllUsesWith(loop.getInductionVar(), idx);

    auto &body = genericOp.getRegion();
    body.takeBody(loop.getRegion());


    blk->eraseArguments(0, blk->getNumArguments());

    for (auto &&[conds, load] : loads) {
        if (stores_map.find(load) != stores_map.end()) {
            // We have a store that represents this load.
            continue;
        }
        auto arg = blk->addArgument(load.getType(), load.getLoc());
        rewriter.replaceOp(load, arg);

    }

    for (auto &&[conds, store] : stores) {
        auto arg = blk->addArgument(store.getValueToStore().getType(), store.getLoc());

        SmallVector<AffineLoadOp> inverted;
        for (auto && [map_load, map_store] : stores_map) {
            if (map_store == store) {
                inverted.push_back(map_load);
            }
        }
        for (size_t i=0; i<inverted.size(); i++) {
            stores_map.erase(inverted[i]);
            auto tmp = inverted[i];
            inverted[i] = nullptr;
            rewriter.replaceOp(tmp, arg);
        }
    }

    SmallVector<Value> toreturn;

    for (auto &&[conds, store] : stores) {
        toreturn.push_back(store.getValueToStore());
        rewriter.eraseOp(store);
    }

    rewriter.eraseOp(blk->getTerminator());
    rewriter.setInsertionPointToEnd(blk);
    rewriter.create<linalg::YieldOp>(loop.getLoc(), toreturn);

    rewriter.eraseOp(loop);
    // return success!
    return success();
  }
};

void RaiseAffineToLinalg::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  // TODO add the existing canonicalization patterns
  //  + subview of an affine apply -> subview
  patterns.insert<AffineForOpRaising>(&getContext());

  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createRaiseAffineToLinalgPass() {
  return std::make_unique<RaiseAffineToLinalg>();
}
} // namespace polygeist
} // namespace mlir
