#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "raise-to-linalg"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace affine;
using namespace linalg;

// Also want to add support for affine.for ( ) { linalg.generic } -> bigger
// linalg.generic Also probably want to try to do { linalg.generc1();
// linalg.generic2(); } -> bigger linalg.generic()

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
    return isLinearInIndex(binop.getLHS(), idx) &&
           isLinearInIndex(binop.getRHS(), idx);
  }
  if (expr.getKind() == AffineExprKind::Mul) {
    auto binop = expr.cast<AffineBinaryOpExpr>();
    return (isLinearInIndex(binop.getLHS(), idx) &&
            !binop.getRHS().isFunctionOfDim(idx)) ||
           (isLinearInIndex(binop.getRHS(), idx) &&
            !binop.getLHS().isFunctionOfDim(idx));
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

AffineExpr shiftDimsDown1(AffineExpr expr, unsigned numDims, unsigned offset) {
  SmallVector<AffineExpr, 4> dims;
  for (unsigned idx = 0; idx < offset; ++idx)
    dims.push_back(getAffineDimExpr(idx, expr.getContext()));
  for (unsigned idx = offset; idx < numDims; ++idx)
    dims.push_back(getAffineDimExpr(idx - 1, expr.getContext()));
  return expr.replaceDimsAndSymbols(dims, {});
}

// This is reducing the number of input dims in expression by 1
AffineMap shiftDimsDown1(AffineMap expr, unsigned numDim, unsigned offset) {
  assert(offset <= expr.getNumDims());
  return AffineMap::get(expr.getNumDims() - 1, expr.getNumSymbols(),
                        llvm::map_to_vector<4>(expr.getResults(),
                                               [&](AffineExpr e) {
                                                 return shiftDimsDown1(
                                                     e, expr.getNumDims(),
                                                     offset);
                                               }),
                        expr.getContext());
}

// Given an affine map `oldmap`, memref `val`, and corresponding input values
// (which are a list of indicies, then symbols), and a set of loop indices `indices` produce
// the following:
//  1. A (potentially new) memref value `newval` which does not have any
//  dependence on `indices`
//     and
//  2. an affine map `newmap` which takes size(indices) values (`indices`) and produces
//  indices into `newval` such that
//     indexing `newval[map(indices)]` produces the same result as indexing the
//     original map.
// check_reduction is set true, when passed from store/linalg.generic's output variable.
// And it is returned true, only if index was not encountered in oldmap operands and check_reduction was set true. 
Value remap_in_affine_dim(bool &legal, OpBuilder &builder, AffineMap oldmap,
                    Value memref_val, Value index, Value bound, int firstNDims, ValueRange oldmap_operands, Value origmemref, bool &check_reduction) {
  assert(oldmap_operands.size() == oldmap.getNumSymbols() + oldmap.getNumDims());
  //Operands which don't correspond to indices
  SmallVector<Value> operands_without_indices;
  ssize_t dimidx = -1;
  for (auto [i, v] : llvm::enumerate(oldmap_operands)) {
    if (v == nullptr) {
      assert(i < firstNDims);
      continue;
    }
    assert(i >= firstNDims);
    if (v != index) {
      // Check if the symbol value is read-only or defined in a scope where it is always visible.
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        // check if it dominates the current scope
        if (ba.getParentBlock()->getParent()->isAncestor(builder.getBlock()->getParent()))
          operands_without_indices.push_back(v);
        else {
          assert(false);
          legal = false;
          return nullptr;
        }
      } else {
        auto op = v.getDefiningOp();
        // check if this dominates the current scope
        if (op->getParentRegion()->isAncestor(builder.getBlock()->getParent())) {
          operands_without_indices.push_back(v);
        } else if (isReadOnly(op)) {
          // if not, check if it is readnone
          // Technically this isn't quite sufficient yet, and does require that the operands to this op are also able to be hoisted,
          // but for now we will assume this
          auto op2 = builder.clone(*op);
          operands_without_indices.push_back(op2->getResult(cast<OpResult>(v).getResultNumber()));
        } else {
          // if so clone it in the right scope
          // otherwise set illegal and don't continue
          assert(false);
          legal = false;
          return nullptr;
        }
      }
    } else
      dimidx = i;
  }
  if((dimidx == -1) && (check_reduction))
    check_reduction = true;
  else 
    check_reduction = false;

  SmallVector<AffineExpr> dimReplacements;
  size_t validSims = 0;
  size_t validDims = 0;
  for (int i=0; i<oldmap.getNumDims(); i++) {
    if (i < firstNDims) {
      assert(i != dimidx);
      dimReplacements.push_back(builder.getAffineDimExpr(validDims));
      validDims++;
    } else if (i == dimidx) {
      dimReplacements.push_back(builder.getAffineDimExpr(validDims));
      validDims++;
    } else {
      // TODO: Why are we using symbol here instead of dim?
      dimReplacements.push_back(builder.getAffineSymbolExpr(validSims));
      validSims++;
    }
  }

  SmallVector<AffineExpr> symReplacements;
  for (int i=0; i<oldmap.getNumSymbols(); i++) {
    if (i + oldmap.getNumDims() == dimidx) {
      symReplacements.push_back(builder.getAffineDimExpr(validDims));
      validDims++;
    } else {
      symReplacements.push_back(builder.getAffineSymbolExpr(validSims));
      validSims++;
    }
  }
  if (validSims != operands_without_indices.size()) {
    llvm::errs() << " oldmap: " << oldmap << "\n";
    llvm::errs() << " dimidx=" << dimidx << "\n";
    llvm::errs() << " index: " << index << "\n";
    llvm::errs() << "  oldmap_operands: size=" << oldmap_operands.size() << "\n";
    for (auto op : oldmap_operands) {
      if (op) {
      llvm::errs() << "  -" << op << " &" << op.getAsOpaquePointer() << "\n";
      } else {
        llvm::errs() << "  -" << "null" << " &nullptr\n";
      }
    }
    llvm::errs() << " validSims: " << validSims << "\n";
    llvm::errs() << " operands_without_indices: size=" << operands_without_indices.size() << "\n";
    for (auto op : operands_without_indices) {
      llvm::errs() << "  -" << op << " &" << op.getAsOpaquePointer() << "\n";
    }
  }
  assert(validSims == operands_without_indices.size());
  auto map2 = oldmap.replaceDimsAndSymbols(dimReplacements, symReplacements, firstNDims+1, operands_without_indices.size());

  SmallVector<Value> idx_sizes;
  for (size_t i=0; i<firstNDims; i++) {
    //memref.dimOp captures the size of the memref
    if (auto submap = origmemref.getDefiningOp<polygeist::SubmapOp>())
      idx_sizes.push_back(submap.getSizes()[i]);
    else
      llvm_unreachable("Won't reach this case");
      //idx_sizes.push_back(builder.create<memref::DimOp>(origmemref.getLoc(), origmemref, i));
  }
  idx_sizes.push_back(bound);

  legal = true;
  SmallVector<int64_t> sizes(idx_sizes.size(), mlir::ShapedType::kDynamic);
  for (auto sz : idx_sizes) {
    // Check if the symbol value is read-only or defined in a scope where it is always visible.
    if (auto ba = dyn_cast<BlockArgument>(sz)) {
      // check if it dominates the current scope
      if (ba.getParentBlock()->getParent()->isAncestor(builder.getBlock()->getParent()))
        operands_without_indices.push_back(sz);
      else {
        llvm::errs() << " value is a non-dominating block arg: " << sz << "\n";
        legal = false;
          assert(false);
        return nullptr;
      }
    } else {
      auto op = sz.getDefiningOp();
      // check if this dominates the current scope
      if (op->getParentRegion()->isAncestor(builder.getBlock()->getParent())) {
        operands_without_indices.push_back(sz);
      } else if (isReadOnly(op)) {
        // if not, check if it is readnone
        // Technically this isn't quite sufficient yet, and does require that the operands to this op are also able to be hoisted,
        // but for now we will assume this
        auto op2 = builder.clone(*op);
        operands_without_indices.push_back(op2->getResult(cast<OpResult>(sz).getResultNumber()));
      } else {
        llvm::errs() << " op is not readonly: " << *op << "\n";
        // if so clone it in the right scope
        // otherwise set illegal and don't continue
        legal = false;
          assert(false);
        return nullptr;
      }
    }
  }
  auto ty = MemRefType::get(sizes, cast<MemRefType>(memref_val.getType()).getElementType());

  return builder.create<polygeist::SubmapOp>(memref_val.getLoc(), ty, memref_val, operands_without_indices, map2);
}

// store A[...]
// val = load A[...]

/*  prevA :
    store A
    val is now prevA
*/

/*

f(%memref )

%memref = ...

affine.for {

    %inp = .. subview %memref [ ... ]

    linalg.generic %inp #map {
      body()
    }
}


->


affine.for j {

    linalg.generic %memref #map2(j) {
      body()
    }
}




#map2 = #map with the indexing done to %inp





%memref = .. subview %memref_base [ ... ]

linalg.generic %[[[memref]]] [[[[#map]]]]([[[[operands]]]]) {
  body()
}

->


output_memref = memref_base
output_map    = subvmap()

 compose 
# uts are memref, map, and operands
# outputs are o
memref[map(operands)] ==== output_memref[output_map(output_operands)]



bas= memref<40x40>

B

u

tput_memref, output_map and output_operands
# possible intermediate is ...

getLinalgArgMap(memref, map, operands to map [e.g. input symbols/dims])
  if memref is alloca/unknown/etc
    return memref/map/operands
  else
    memref = subview memref_base[map2(operands2)]

    return memref_base   and a new output_map such that
      memref_base[output_map(output_operands)] === memref[map(operands)]





*/

// Suppose we have a memref expression E=input[affine.map(operands)]
//     if input = memref.subview A[starts, offsets]
//    can we rewrite E as A[affine.map2(operands2)]
//    We update lgMap and lgOperands in place with this coresponding map2 and
//    operands2
LogicalResult getLinalgArgMap(Operation *loop, Value &input, AffineMap &lgMap,
                              SmallVector<Value> &lgOperands) {
  OpBuilder builder(loop->getContext());

  while (Operation *defOp = input.getDefiningOp()) {

    assert(lgOperands.size() == lgMap.getNumSymbols() + lgMap.getNumDims());
    // If the input is defined outside of the loop, we are finished.
    if (!loop->isAncestor(defOp))
      break;

    if (auto SM = dyn_cast<polygeist::SubmapOp>(defOp)) {
      auto submap = SM.getMap();

      //TODO: Do we achieve anything with this compose?
      //As lgMap in our case is 1 to 1 identity map
      auto composeMap = submap.compose(lgMap);

      SmallVector<Value> operands0;

      // First the dims
      for (size_t i = 0; i < lgMap.getNumDims(); i++)
        operands0.push_back(lgOperands[i]);

      // Then the symbols of submap
      for (size_t i = 0; i < submap.getNumSymbols(); i++)
        operands0.push_back(SM.getSymbols()[i]);

      // Then the symbols of lgMap
      for (size_t i = 0; i < lgMap.getNumSymbols(); i++)
        operands0.push_back(lgOperands[i + lgMap.getNumDims()]);

      lgMap = composeMap;
      lgOperands = operands0;
      input = SM.getMemref();
      assert(lgOperands.size() == lgMap.getNumSymbols() + lgMap.getNumDims());
      continue;
    }

    //if (auto SV = dyn_cast<memref::SubViewOp>(defOp)) {

    //  // TODO update map with the new indexing from here

    //  // Create affine map
    //  //   i. Track number of running dims and symbols
    //  //  ii. shift dims and symbols to generate shifted expressions.
    //  // Extract corresponding operands
    //  // Use affineMap::get with numOperands and numSymbols along with shifted
    //  // expressions to get a map. Use affine map simplify to simplify this

    //  SmallVector<AffineExpr> startExprs;
    //  SmallVector<AffineExpr> strideExprs;
    //  SmallVector<Value> dimOperands;
    //  SmallVector<Value> symOperands;
    //  for (auto &&[first, second] : llvm::zip(SV.getOffsets(), SV.getStrides())) {
    //    for (auto &&[index, val] : llvm::enumerate(SmallVector<Value>({first, second}))) {
    //      auto &exprOutput = (index == 0) ? startExprs : strideExprs;
    //      // Only support constants, symbols, or affine apply as offsets
    //      if (auto cop = val.getDefiningOp<arith::ConstantIntOp>()) {
    //        exprOutput.push_back(builder.getAffineConstantExpr(cop.value()));
    //        continue;
    //      } else if (auto cop = val.getDefiningOp<arith::ConstantIndexOp>()) {
    //        exprOutput.push_back(builder.getAffineConstantExpr(cop.value()));
    //        continue;
    //      }
    //      if (auto ba = dyn_cast<BlockArgument>(val)) {
    //        Block *parentBlock = ba.getOwner();
    //        if (isa<AffineForOp, AffineParallelOp>(parentBlock->getParentOp())) {
    //          exprOutput.push_back(
    //              builder.getAffineDimExpr(dimOperands.size()));
    //          dimOperands.push_back(ba);
    //          continue;

    //        }
    //      }

    //      auto valOp = val.getDefiningOp();
    //      // Defined outside loop, consider it a symbol [for now]
    //      //if (!valOp || loop->isAncestor(defOp)) {
    //      if (valOp&&!loop->isAncestor(defOp)) {
    //        exprOutput.push_back(
    //            builder.getAffineSymbolExpr(symOperands.size()));
    //        symOperands.push_back(val);
    //        continue;
    //      }

    //      //TODO: Maybe it's a case to add, but are we sure we need it for starts and offsets
    //      // and not for operands
    //      if (auto apply = dyn_cast<AffineApplyOp>(valOp)) {
    //        auto map = apply.getAffineMap();
    //        auto *scope = affine::getAffineScope(valOp)->getParentOp();
    //        DominanceInfo DI(scope);
    //        auto map_operands = apply.getOperands();
    //        //fully2ComposeAffineMapAndOperands(builder, &map, &map_operands, DI);
  //// Instead of using loop step we are using 1 (Assumption as the stride size)
    //        auto newexpr = map.shiftDims(dimOperands.size())
    //                           .shiftSymbols(symOperands.size());

    //        for (auto expr : newexpr.getResults()) {
    //          exprOutput.push_back(expr);
    //        }

    //        for (size_t i = 0; i < map.getNumDims(); i++)
    //          dimOperands.push_back(apply.getOperands()[i]);

    //        for (size_t i = 0; i < map.getNumSymbols(); i++)
    //          symOperands.push_back(apply.getOperands()[i + map.getNumDims()]);

    //        continue;
    //      }

    //      //return failure();
    //    }
    //  }

    //  SmallVector<AffineExpr> inputExprs;
    //  for (auto expr : lgMap.shiftDims(dimOperands.size())
    //                       .shiftSymbols(symOperands.size()).getResults()) {
    //    inputExprs.push_back(expr);
    //  }
    //  for (size_t i = 0; i < lgMap.getNumDims(); i++)
    //    dimOperands.push_back(lgOperands[i]);

    //  for (size_t i = 0; i < lgMap.getNumSymbols(); i++)
    //    symOperands.push_back(lgOperands[i + lgMap.getNumDims()]);


    //  SmallVector<AffineExpr> mergedExprs;
    //  for (auto && [start, stride, idx] :
    //       llvm::zip(startExprs, strideExprs, inputExprs)) {
    //    mergedExprs.push_back(start + idx * stride);
    //  }

    //  lgMap =
    //      AffineMap::get(dimOperands.size(), symOperands.size(), mergedExprs, loop->getContext());
    //  lgOperands.clear();
    //  lgOperands.insert(lgOperands.begin(), dimOperands.begin(), dimOperands.end());
    //  lgOperands.insert(lgOperands.begin()+lgOperands.size(), symOperands.begin(), symOperands.end());
    //  input = SV.getSource();
    //  break;
    //}

    //return failure();
  }
  assert(lgOperands.size() == lgMap.getNumSymbols() + lgMap.getNumDims());
  return success();
}

struct AffineForOpRaising : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp loop,
                                PatternRewriter &rewriter) const final {
    
    auto module = loop->getParentOfType<ModuleOp>();

    // Don't handle accumulations in registers for the moment, we can have
    // a separate pattern move them into memref's
    if (loop.getNumResults() != 0) {
      return failure();
    }

    SmallVector<std::pair<std::vector<Condition>, AffineLoadOp>> loads;
    SmallVector<std::pair<std::vector<Condition>, AffineStoreOp>> stores;
    SmallVector<std::pair<std::vector<Condition>, GenericOp>> linalgGenerics;
    bool check_reduction;
    
    // TODO Also collect all the linalg generics!

    // Check that the only operations within the region are either:
    //      affine.load, affine.store, affine.if, affine.yield
    // Additionally, for each load/store, remember what conditions are
    // required for that load or store to execute.
    auto result = loop->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == loop)
        return WalkResult::advance();
      // TODO extend this, any non-memory operation is also legal here.
      // mul, add, etc (we can just check propety)
      if (isa<AffineYieldOp, AffineIfOp>(op)) {
        return WalkResult::advance();
      }
      if (isa<AffineLoadOp, AffineStoreOp>(op) || isa<GenericOp>(op)) {
        Operation *cur = op->getParentOp();
        std::vector<Condition> conditions;
        while (cur != loop) {
          auto ifstmt = dyn_cast<AffineIfOp>(cur);
          if (!ifstmt) {
            return WalkResult::interrupt();
          }
          bool ifTrue =
              ifstmt.getThenRegion().isAncestor(cur->getParentRegion());
          conditions.emplace_back(ifTrue, ifstmt);
          cur = ifstmt->getParentOp();
        }
        if (auto linalgGeneric = dyn_cast<GenericOp>(op)) {
          linalgGenerics.emplace_back(conditions, linalgGeneric);
        } else if (auto load = dyn_cast<AffineLoadOp>(op)) {
          loads.emplace_back(conditions, load);
        } else {
          auto store = cast<AffineStoreOp>(op);
          stores.emplace_back(conditions, store);
        }
        return WalkResult::advance();
      }
      // IsReadNone takes care of apply and subview too?
      if (isReadNone(op)) {
        return WalkResult::advance();
      }
      return WalkResult::interrupt();
    });

    if (result.wasInterrupted())
      return failure();

    DominanceInfo DI(loop);

    // Check that all of the stores do not alias the loaded values (otherwise we
    // could get an incorrect result)
    // TODO we can extend this and handle things like reductions, but we're
    // going to start easy for now
    // TODO
    DenseMap<AffineLoadOp, AffineStoreOp> stores_map;
    for (auto &&[_, store] : stores) {
      for (auto &&[_, load] : loads) {
        if (mayAlias(load.getMemref(), store.getMemref())) {
          // We have one exception in this case -- if the load and store are
          // from the exact same location, it is permitted.
          if (load.getMemref() == store.getMemref() &&
              load.getAffineMap() == store.getAffineMap() &&
              load.getIndices() == store.getIndices() &&
              DI.dominates((Operation *)load, (Operation *)store)) {
            // Example case where load does not dominate stores - if the load
            // was conditional. Or, store followed by load? Q. Can't we still
            // overlook the aliasing?
            stores_map[load] = store;
            continue;
          }
          return failure();
        }
      }
      for (auto &&[_, store2] : stores) {
        if (store == store2)
          continue;
        if (mayAlias(store.getMemref(), store2.getMemref())) {
          return failure();
        }
      }
    }
    // Check that any other loads / stores do not alias with any linalg generics
    // We're going to need to upgrade the defn of mayAlias for subviews (aka
    // mayAlias(subview, x) -> mayAlias(operand(subview), x))

    SmallVector<Value> inputs, outputs;
    SmallVector<AffineMap> affineMaps;
    SmallVector<AffineMap> indexingMaps;

    // if (loop.getStep() != 1) {
    //     return failure();
    // }

    // our remapper currently assumes 0 start to bound.
    if (!loop.hasConstantLowerBound() /*|| loop.getConstantLowerBound() != 0*/) {
      return failure();
    }

    // compute this correctly later.
    auto ubMap = loop.getUpperBoundMap();
    auto ubOperands = loop.getUpperBoundOperands();
    if (!ubMap || ubMap.getNumResults() != 1)
      return failure();

    // Retrieve the lower bound
    auto lbMap = loop.getLowerBoundMap();
    auto lbOperands = loop.getLowerBoundOperands();
    if (!lbMap || lbMap.getNumResults() != 1)
      return failure();

    auto ub = loop.getSingleUpperBound();
    if (!ub)
      return failure();

    auto lb = loop.getSingleLowerBound();
    if (!lb)
      return failure();

    if (!loop.hasConstantUpperBound()) {
      return failure();
    }

    // Retrieve the step size
    int64_t step = loop.getStep();

    // Get the single result expressions
    AffineExpr ubExpr = ubMap.getResult(0);
    auto ubValue =
        rewriter.create<AffineApplyOp>(loop.getLoc(), ubMap, ubOperands);

    AffineExpr lbExpr = lbMap.getResult(0);
    auto lbValue =
        rewriter.create<AffineApplyOp>(loop.getLoc(), lbMap, lbOperands);

    //// Ensure the bounds are constant expressions
    auto ubConst = ubExpr.dyn_cast<AffineConstantExpr>();
    auto lbConst = lbExpr.dyn_cast<AffineConstantExpr>();
    if (!ubConst || !lbConst)
      return failure();

    // Compute the loop size
    // int64_t loopSize = ubConst.getValue() - lbConst.getValue();
    auto loopSize = rewriter.create<SubIOp>(loop.getLoc(), ubValue, lbValue);

    // Value loopSize = rewriter.create<arith::ConstantIndexOp>(loop.getLoc(),
    // loop.getConstantUpperBound());//rewriter.create<arith::SubIOp>(loop.getLoc(),
    // *ub, *lb);


    for (auto &&[conds, lg] : linalgGenerics) {

      // This captures the indexing map attribute from the linalg.generic being
      // processed
      ArrayAttr indexingMapsAttr = lg.getIndexingMaps();

      int idx = 0;
      // Iterate over input arguments
      for (const Value input : lg.getInputs()) {
        // Is this needed?
        if (conds.size() != 0)
          return failure();

        // TODO: Implement this
        // lgMap comes from offset of memref.subview,
        // lgOperands comes from operands of memref.subview

        const AffineMap lgMap0 = cast<AffineMapAttr>(indexingMapsAttr[idx]).getAffineMap();
        AffineMap lgMap = lgMap0;
        SmallVector<Value> lgOperands;
        for (int i=0; i<lgMap.getNumDims(); i++) {
          lgOperands.push_back(nullptr);
        }

        Value lgMemref = input;
        
        // At input, this contains, current input (i.e. probably a subview)
        // an  lgMap which is obtained from LG's indexing map for corresponding input
        // lgOperands contains current input (i.e probably a subview)

        // Gives output ...
        
        assert(lgOperands.size() == lgMap.getNumSymbols() + lgMap.getNumDims());
        auto result = getLinalgArgMap(loop, lgMemref, lgMap, lgOperands);

        if (!result.succeeded())
          return failure();

        bool legal = true;
        
        // Takes input's/output's, affineMap of load/store (here lgMap ?), 
        // induction variable corresponding to the loop
        // Memref corresponding the the memory accessed (in this case subview ?)
        // loopSize, lower and upper bounds
        // Get operands for load/store (here ?) to find dependent dim

        // Gives output newMemref which is a subviewOp,
        // newAffineMap which is the LG's indexing map corresponding this inp/output
        
        // This takes load and store maps and then creates affine.apply+subview+linalg.generic
        // For this case: LG within ForOp -
        // Inputs should be : load map extracted from subviewOp
        // Returns LG with indexingMap and subview  with affine.apply - which are correct 
        
        //TODO: Or is it num dims?
        //size_t firstNDims = lgMap.getResults().size();
        size_t firstNDims = lgMap.getNumDims();
        check_reduction = false;
        auto newMemref = remap_in_affine_dim(
            legal, rewriter, lgMap, lgMemref, loop.getInductionVar(), loopSize,
            firstNDims, ValueRange(lgOperands), input, check_reduction);
        if (!legal)
          return failure();

        auto newAffineMap = rewriter.getMultiDimIdentityMap(firstNDims+1);
        
        // TODO: need to mergre previous indexing maps and new affine maps
        affineMaps.push_back(newAffineMap);
        inputs.push_back(newMemref);
        idx++;
      }

      // Iterate over output arguments
      for (const Value output : lg.getOutputs()) {
        // Is this needed?
        if (conds.size() != 0)
          return failure();

        const AffineMap lgMap0 = cast<AffineMapAttr>(indexingMapsAttr[idx]).getAffineMap();
        AffineMap lgMap = lgMap0;
        
        SmallVector<Value> lgOperands;
        for (int i=0; i<lgMap.getNumDims(); i++) {
          lgOperands.push_back(nullptr);
        }
        Value lgMemref = output;
        
        auto result = getLinalgArgMap(loop, lgMemref, lgMap, lgOperands);

        if (!result.succeeded())
          return failure();

        bool legal = true;

        size_t firstNDims = lgMap.getNumDims();
        check_reduction = true;
        auto newMemref = remap_in_affine_dim(
            legal, rewriter, lgMap, lgMemref, loop.getInductionVar(), loopSize, firstNDims, ValueRange(lgOperands), output, check_reduction);
        if (!legal)
          return failure();

        auto newAffineMap = rewriter.getMultiDimIdentityMap(firstNDims+1);
        // TODO: need to merge previous indexing maps and new affine maps
        affineMaps.push_back(newAffineMap);
        outputs.push_back(newMemref);
      }
    }

    // current spec is going to be indexed off of the loop var in isolation
    for (auto &&[conds, load] : loads) {
      // Only support unconditional loads for the moment
      if (conds.size() != 0)
        return failure();

      if (stores_map.find(load) != stores_map.end()) {
        // We have a store that represents this load.
        continue;
      }

      size_t firstNDims = 0;
      bool legal = true;

      check_reduction = false;
      auto newMemref = remap_in_affine_dim(
          legal, rewriter, load.getAffineMap(), load.getMemref(),
          loop.getInductionVar(), loopSize, firstNDims,
          load.getMapOperands(), load.getMemref(), check_reduction);

      if (!legal)
        return failure();

      auto newAffineMap = rewriter.getMultiDimIdentityMap(firstNDims+1);
      affineMaps.push_back(newAffineMap);
      inputs.push_back(newMemref);
    }
    // TODO Push all of the inputs to the linalg generics (modifying maps as
    // needed)

    //SmallVector<Value> outputs;
    // Store we may need to reindex into a splat potentially later, but for now
    // we'll be lazy
    for (auto &&[conds, store] : stores) {
      // Only support unconditional loads for the moment
      if (conds.size() != 0)
        return failure();

      bool legal = true;

      size_t firstNDims = 0;
      
      check_reduction = true;
      auto newMemref = remap_in_affine_dim(
          legal, rewriter, store.getAffineMap(), store.getMemref(),
          loop.getInductionVar(), loopSize, firstNDims,
          store.getMapOperands(), store.getMemref(), check_reduction);

      if (!legal) {
        return failure();
      }

      auto newAffineMap = rewriter.getMultiDimIdentityMap(firstNDims+1);
      affineMaps.push_back(newAffineMap);
      outputs.push_back(newMemref);
    }
    // TODO Push all of the outputs to the linalg generics

    // TODO presently  if linalg generic exists, assert there are no load/stores
    if ((linalgGenerics.size() > 0) &&
          ((loads.size() != 0) || (stores.size() != 0))) {
      assert(false);
      return failure();
    }

    // TODO assert only zero or one linalg generic exists
    if (!(linalgGenerics.size() == 1 || linalgGenerics.size() == 0)) {
      //assert(false);
      return failure();
    }

    SmallVector<utils::IteratorType> iteratorTypes;
    // TODO if linalg generic exists, make this iterator type prepend to the
    // existing iterators

    //TODO: Just store check is not sufficient, there has to be a check for 
    //bool is_parallel = stores_map.size() == 0;
    // TODO determine if linalg generic, whether to create parallel or reduction by looking at memory patterns of maps

    if (linalgGenerics.size() == 1) {
      // determine whether now we write to ourselves
    }

    iteratorTypes.push_back(check_reduction
                                ? utils::IteratorType::reduction
                                : utils::IteratorType::parallel);

    if (linalgGenerics.size() == 1) {
      for (auto attr : linalgGenerics[0].second.getIteratorTypesArray())
          iteratorTypes.push_back(attr);
    }

    StringAttr empty = StringAttr::get(loop.getContext());
    auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
        loop.getLoc(), TypeRange(), inputs, outputs, affineMaps, iteratorTypes,
        empty, empty);

    // TODO if doing the linalg generic case, ignore a lot of the below and
    // instead of injecting the old body of the affine.for, move the inner
    // linalg.generic body and also add a new induction variable
    auto blk = &*loop.getRegion().begin();
    rewriter.setInsertionPointToStart(blk);

    // This index will replace the use of the affine index
    auto idx = rewriter.create<linalg::IndexOp>(loop.getLoc(),
                                                rewriter.getIndexAttr(0));
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
      auto arg =
          blk->addArgument(store.getValueToStore().getType(), store.getLoc());

      SmallVector<AffineLoadOp> inverted;
      for (auto &&[map_load, map_store] : stores_map) {
        if (map_store == store) {
          inverted.push_back(map_load);
        }
      }
      for (size_t i = 0; i < inverted.size(); i++) {
        stores_map.erase(inverted[i]);
        auto tmp = inverted[i];
        inverted[i] = nullptr;
        rewriter.replaceOp(tmp, arg);
      }
    }

    SmallVector<Value> toreturn;

    for (auto genPair : linalgGenerics) {
      auto genOp = genPair.second;
      auto &genBlock = genOp->getRegion(0).front();
      auto term = genBlock.getTerminator();
      mlir::IRMapping map;
      for (auto arg : genBlock.getArguments()) {
        auto arg2 =
            blk->addArgument(arg.getType(), arg.getLoc());
        map.map(arg, arg2);
      }
      for (auto &op : genBlock.without_terminator()) {
        rewriter.clone(op, map);
      }
      for (auto op : term->getOperands()) {
        toreturn.push_back(map.lookup(op));
      }
      //llvm::errs() << genOp->getParentOfType<func::FuncOp>() << "\n";
      rewriter.eraseOp(genOp);
    }

    for (auto &&[conds, store] : stores) {
      toreturn.push_back(store.getValueToStore());
      rewriter.eraseOp(store);
    }

    rewriter.eraseOp(blk->getTerminator());
    rewriter.setInsertionPointToEnd(blk);
    rewriter.create<linalg::YieldOp>(loop.getLoc(), toreturn);

    auto func = loop->getParentOfType<func::FuncOp>();
    rewriter.eraseOp(loop);
    // return success!
    return success();
  }
};

// struct RemoveIterArgs : public OpRewritePattern<scf::ForOp> {
//   using OpRewritePattern<scf::ForOp>::OpRewritePattern;
//   LogicalResult matchAndRewrite(scf::ForOp forOp,
//                                 PatternRewriter &rewriter) const override {
//     if (!forOp.getRegion().hasOneBlock())
//       return failure();
//     unsigned numIterArgs = forOp.getNumRegionIterArgs();
//     auto loc = forOp->getLoc();
//     bool changed = false;
//     llvm::SetVector<unsigned> removed;
//     llvm::MapVector<unsigned, Value> steps;
//     auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
//     for (unsigned i = 0; i < numIterArgs; i++) {
//       auto ba = forOp.getRegionIterArgs()[i];
//       auto init = forOp.getInits()[i];
//       auto next = yield->getOperand(i);

//       auto increment = next.getDefiningOp<arith::AddIOp>();
//       if (!increment)
//         continue;

//       Value step = nullptr;
//       if (increment.getLhs() == ba) {
//         step = increment.getRhs();
//       } else {
//         step = increment.getLhs();
//       }
//       if (!step)
//         continue;

//       // If it dominates the loop entry
//       if (!step.getParentRegion()->isProperAncestor(&forOp.getRegion()))
//         continue;

//       rewriter.setInsertionPointToStart(forOp.getBody());
//       Value iterNum = rewriter.create<arith::SubIOp>(
//           loc, forOp.getInductionVar(), forOp.getLowerBound());
//       iterNum = rewriter.create<arith::DivSIOp>(loc, iterNum, forOp.getStep());

//       Value replacementIV = rewriter.create<arith::MulIOp>(loc, iterNum, step);
//       replacementIV = rewriter.create<arith::AddIOp>(loc, replacementIV, init);

//       rewriter.replaceAllUsesWith(ba, replacementIV);

//       removed.insert(i);
//       steps.insert({i, step});
//       changed = true;
//     }

//     if (!changed)
//       return failure();

//     SmallVector<Value> newInits;
//     for (unsigned i = 0; i < numIterArgs; i++)
//       if (!removed.contains(i))
//         newInits.push_back(forOp.getInits()[i]);

//     rewriter.setInsertionPoint(forOp);
//     auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
//                                                 forOp.getUpperBound(),
//                                                 forOp.getStep(), newInits);
//     if (!newForOp.getRegion().empty())
//       newForOp.getRegion().front().erase();
//     assert(newForOp.getRegion().empty());
//     rewriter.inlineRegionBefore(forOp.getRegion(), newForOp.getRegion(),
//                                 newForOp.getRegion().begin());

//     SmallVector<Value> newYields;
//     for (unsigned i = 0; i < numIterArgs; i++)
//       if (!removed.contains(i))
//         newYields.push_back(yield->getOperand(i));

//     rewriter.setInsertionPoint(yield);
//     rewriter.replaceOpWithNewOp<scf::YieldOp>(yield, newYields);

//     llvm::BitVector toDelete(numIterArgs + 1);
//     for (unsigned i = 0; i < numIterArgs; i++)
//       if (removed.contains(i))
//         toDelete[i + 1] = true;
//     newForOp.getBody()->eraseArguments(toDelete);

//     rewriter.setInsertionPoint(newForOp);
//     unsigned curNewRes = 0;
//     for (unsigned i = 0; i < numIterArgs; i++) {
//       auto result = forOp->getResult(i);
//       if (removed.contains(i)) {
//         if (result.use_empty())
//           continue;

//         rewriter.setInsertionPointToStart(forOp.getBody());
//         Value iterNum = rewriter.create<arith::SubIOp>(
//             loc, forOp.getUpperBound(), forOp.getLowerBound());
//         iterNum =
//             rewriter.create<arith::DivSIOp>(loc, iterNum, forOp.getStep());

//         Value afterLoop =
//             rewriter.create<arith::MulIOp>(loc, iterNum, steps[i]);
//         afterLoop =
//             rewriter.create<arith::AddIOp>(loc, afterLoop, forOp.getInits()[i]);

//         rewriter.replaceAllUsesWith(result, afterLoop);
//       } else {
//         rewriter.replaceAllUsesWith(result, newForOp->getResult(curNewRes++));
//       }
//     }

//     rewriter.eraseOp(forOp);

//     return success();
//   }
// };

namespace {
struct RaiseAffineToLinalg
    : public AffineRaiseToLinalgBase<RaiseAffineToLinalg> {

  std::shared_ptr<const FrozenRewritePatternSet> patterns;

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    //owningPatterns.insert<RemoveIterArgs>(&getContext());
    owningPatterns.insert<AffineForOpRaising>(&getContext());

    patterns = std::make_shared<FrozenRewritePatternSet>(
        std::move(owningPatterns));
    return success();
  }
  void runOnOperation() override {
    GreedyRewriteConfig config;
    (void)applyPatternsAndFoldGreedily(getOperation(), *patterns, config);
  }
};
} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createRaiseAffineToLinalgPass() {
  return std::make_unique<RaiseAffineToLinalg>();
}
} // namespace polygeist
} // namespace mlir
