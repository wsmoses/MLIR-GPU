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

#define DEBUG_TYPE "remove-scf-iter-args"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace scf;
using namespace affine;

struct RemoveSCFIterArgs : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    
    ModuleOp module = forOp->getParentOfType<ModuleOp>();
    if (!forOp.getRegion().hasOneBlock())
      return failure();
    unsigned numIterArgs = forOp.getNumRegionIterArgs();
    auto loc = forOp->getLoc();
    bool changed = false;
    llvm::SetVector<unsigned> removed;
    llvm::MapVector<unsigned, Value> steps;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (unsigned i = 0; i < numIterArgs; i++) {
      auto ba = forOp.getRegionIterArgs()[i];
      auto init = forOp.getInits()[i];
      auto lastOp = yieldOp->getOperand(i);

      //General Case(TODO):
      //ALGo:
      // 1. Create an alloca(stack) variable
      //    How to know it's dims? It should be based on number of reduction loops
      // 2. Initialize it with init value just outside the for loop if init value is non-zero
      // 3. memref.load that value in the for loop
      // 4. Replace all the uses of the iter_arg with the loaded value
      // 5. Add a memref.store for the value to be yielded
      // 6. Replace all uses of for-loops yielded value with a single inserted memref.load 
      //Special case:
      //ALGo:
      //Optimize away memref.store and memref.load, if the only users of memref.load are memref.store (can use affine-scalrep pass for that ? No it does store to load forwarding)
      //What we need is forwarding of local store to final store and deleting the intermediate alloca created. This is only possible if the user of alloca is a storeOp.
      // 1. Identify the single store of the for loop result
      // 2. Initialize it with iter arg init, outside the for loop. (TODO)
      // 3. Do a load from the memref
      // 4. move the store to memref inside the loop.

      auto result = forOp.getResult(i);
      if(result.hasOneUse()) {
        auto storeOp = dyn_cast<memref::StoreOp>(*result.getUsers().begin());
        if(storeOp)
        {
          {
            rewriter.setInsertionPointToStart(forOp.getBody());
            auto memrefLoad = rewriter.create<memref::LoadOp>(
                forOp.getLoc(), storeOp.getMemref(), storeOp.getIndices());
            rewriter.replaceAllUsesWith(ba, memrefLoad.getResult());
          }
          {
            rewriter.setInsertionPoint(yieldOp);
            rewriter.create<memref::StoreOp>(forOp.getLoc(), lastOp, storeOp.getMemref(),
                                           storeOp.getIndices());
            storeOp.erase();                   
          }
        }
        else{
          return failure();
        }
      }
      //else{
      //  alloca = rewriter.create<memref::AllocaOp>(
      //        forOp.getLoc(), MemRefType::get(ArrayRef<int64_t>(), forOp.getType()),
      //        ValueRange());
      //  //Skipping init for now


      //  auto memrefLoad = rewriter.create<memref::LoadOp>(
      //      forOp.getLoc(), alloca.getMemref(), op.getIndices());
      //  rewriter.replaceOp(op, memrefLoad.getResult());
      
      //  rewriter.create<memref::StoreOp>(forOp.getLoc(), lastOp, alloca,
      //                                   forOp.getBody()->getArguments());

      //  rewriter.replaceAllUsesWith(result,)
      //}

      rewriter.setInsertionPointToStart(forOp.getBody());
      //rewriter.replaceAllUsesWith(ba, replacementIV);
      changed = true;
    }

    if (!changed)
      return failure();

    rewriter.setInsertionPoint(forOp);
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                forOp.getUpperBound(),
                                                forOp.getStep());
    if (!newForOp.getRegion().empty())
      newForOp.getRegion().front().erase();
    assert(newForOp.getRegion().empty());
    rewriter.inlineRegionBefore(forOp.getRegion(), newForOp.getRegion(),
                                newForOp.getRegion().begin());

    //Delete region args
    llvm::BitVector toDelete(numIterArgs + 1);
    for (unsigned i = 0; i < numIterArgs; i++)
        toDelete[i + 1] = true;
    newForOp.getBody()->eraseArguments(toDelete);

    SmallVector<Value> newYields;
    {
      ValueRange empty;
      rewriter.setInsertionPoint(yieldOp);
      auto newYieldOp = rewriter.create<scf::YieldOp>(loc);
      //rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYieldOp);
      rewriter.eraseOp(yieldOp);
    }

    rewriter.setInsertionPoint(newForOp);
    rewriter.eraseOp(forOp);

    return success();
  }
};

struct RemoveAffineIterArgs : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    
    ModuleOp module = forOp->getParentOfType<ModuleOp>();
    if (!forOp.getRegion().hasOneBlock())
      return failure();
    unsigned numIterArgs = forOp.getNumRegionIterArgs();
    auto loc = forOp->getLoc();
    bool changed = false;
    llvm::SetVector<unsigned> removed;
    llvm::MapVector<unsigned, Value> steps;
    auto yieldOp = cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());
    for (unsigned i = 0; i < numIterArgs; i++) {
      auto ba = forOp.getRegionIterArgs()[i];
      auto init = forOp.getInits()[i];
      auto lastOp = yieldOp->getOperand(i);

      //General Case(TODO):
      //ALGo:
      // 1. Create an alloca(stack) variable
      //    How to know it's dims? It should be based on number of reduction loops
      // 2. Initialize it with init value just outside the for loop if init value is non-zero
      // 3. memref.load that value in the for loop
      // 4. Replace all the uses of the iter_arg with the loaded value
      // 5. Add a memref.store for the value to be yielded
      // 6. Replace all uses of for-loops yielded value with a single inserted memref.load 
      //Special case:
      //ALGo:
      //Optimize away memref.store and memref.load, if the only users of memref.load are memref.store (can use affine-scalrep pass for that ? No it does store to load forwarding)
      //What we need is forwarding of local store to final store and deleting the intermediate alloca created. This is only possible if the user of alloca is a storeOp.
      // 1. Identify the single store of the for loop result
      // 2. Initialize it with iter arg init, outside the for loop. (TODO)
      // 3. Do a load from the memref
      // 4. move the store to memref inside the loop.

      auto result = forOp.getResult(i);
      if(result.hasOneUse()) {
        auto storeOp = dyn_cast<affine::AffineStoreOp>(*result.getUsers().begin());
        if(storeOp)
        {
          {
            rewriter.setInsertionPointToStart(forOp.getBody());
            auto memrefLoad = rewriter.create<affine::AffineLoadOp>(
                forOp.getLoc(), storeOp.getMemref(), storeOp.getMap(), storeOp.getMapOperands());
            rewriter.replaceAllUsesWith(ba, memrefLoad.getResult());
          }
          {
            rewriter.setInsertionPoint(yieldOp);
            rewriter.create<affine::AffineStoreOp>(forOp.getLoc(), lastOp, storeOp.getMemref(),
                                           storeOp.getMap(), storeOp.getMapOperands());
            storeOp.erase();                   
          }
        }
        else{
          return failure();
        }
      }
      //else{
      //  alloca = rewriter.create<memref::AllocaOp>(
      //        forOp.getLoc(), MemRefType::get(ArrayRef<int64_t>(), forOp.getType()),
      //        ValueRange());
      //  //Skipping init for now


      //  auto memrefLoad = rewriter.create<affine::AffineLoadOp>(
      //      forOp.getLoc(), alloca.getMemref(), op.getIndices());
      //  rewriter.replaceOp(op, memrefLoad.getResult());
      
      //  rewriter.create<affine::AffineStoreOp>(forOp.getLoc(), lastOp, alloca,
      //                                   forOp.getBody()->getArguments());

      //  rewriter.replaceAllUsesWith(result,)
      //}

      rewriter.setInsertionPointToStart(forOp.getBody());
      //rewriter.replaceAllUsesWith(ba, replacementIV);
      changed = true;
    }

    if (!changed)
      return failure();

    rewriter.setInsertionPoint(forOp);
    auto newForOp = rewriter.create<affine::AffineForOp>(loc, forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
                                                forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
                                                forOp.getStep());
    
    if (!newForOp.getRegion().empty())
      newForOp.getRegion().front().erase();
    assert(newForOp.getRegion().empty());
    rewriter.inlineRegionBefore(forOp.getRegion(), newForOp.getRegion(),
                                newForOp.getRegion().begin());

    //Delete region args
    llvm::BitVector toDelete(numIterArgs + 1);
    for (unsigned i = 0; i < numIterArgs; i++)
        toDelete[i + 1] = true;
    newForOp.getBody()->eraseArguments(toDelete);

    SmallVector<Value> newYields;
    {
      ValueRange empty;
      rewriter.setInsertionPoint(yieldOp);
      auto newYieldOp = rewriter.create<affine::AffineYieldOp>(loc);
      //rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(yieldOp, newYieldOp);
      rewriter.eraseOp(yieldOp);
    }

    rewriter.setInsertionPoint(newForOp);
    rewriter.eraseOp(forOp);

    return success();
  }
};

namespace {
struct RemoveIterArgs
    : public RemoveIterArgsBase<RemoveIterArgs> {

  void runOnOperation() override {
    GreedyRewriteConfig config;
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    patterns.insert<RemoveSCFIterArgs>(patterns.getContext());
    patterns.insert<RemoveAffineIterArgs>(patterns.getContext());
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                          config))) {
    signalPassFailure();
    return;
    }
  }
};
} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createRemoveIterArgsPass() {
  return std::make_unique<RemoveIterArgs>();
}
} // namespace polygeist
} // namespace mlir