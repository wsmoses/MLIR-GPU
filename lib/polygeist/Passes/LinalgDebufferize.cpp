#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

#define DEBUG_TYPE "linalg-debufferize"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace affine;
using namespace linalg;
using namespace tensor;
using namespace bufferization;

std::vector<Operation *> getSortedUsers(Operation *op) {
  if(!op) return {};

  // Find the parent function
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) return {};

  //Map to store order of operations
  llvm::DenseMap<Operation *, size_t> opOrder;
  size_t order = 0;

  funcOp.walk([&](Operation *curOp) {
     opOrder[curOp] = order++;
  });

  std::vector<Operation *> sortedUsers(op->getUsers().begin(), op->getUsers().end());

  std::sort(sortedUsers.begin(), sortedUsers.end(),
    [&](Operation *a, Operation *b) {
      return opOrder[a] < opOrder[b];
    }
  );

  return sortedUsers;
}

struct LinalgDebufferization : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final {
    
    auto module = funcOp->getParentOfType<ModuleOp>();

    SmallVector<Operation*> opsToDelete;
    llvm::SmallPtrSet<Operation*, 16> opsToDeleteSet;
    //Tracks both old linalg.generics and linalg.generics with repeated values in ins and outs
    llvm::SmallPtrSet<Operation*, 16> processedGenericOps;

    LogicalResult passResult = success();
    funcOp.walk([&](mlir::memref::AllocaOp allocaOp) -> WalkResult {
        auto module = allocaOp->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointAfter(allocaOp);
        auto tensorType = RankedTensorType::get(allocaOp.getType().getShape(), allocaOp.getType().getElementType());

        //Check to see if only linalg.generic are users of the allocaOp for now.
        //TODO: Extend this
        if(!llvm::all_of(allocaOp->getUsers(),[](Operation *op) {
           return isa<linalg::GenericOp>(op);
        })){
          passResult = failure();
          return WalkResult::interrupt();
        }
      
        //auto emptyTensor = rewriter.create<tensor::EmptyOp>(allocaOp.getLoc(),allocaOp.getType().getShape(), allocaOp.getType().getElementType());
        auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
                                allocaOp.getLoc(),
                                tensorType,
                                allocaOp);
        Value currentTensor = toTensorOp;

        auto sortedUsers = getSortedUsers(allocaOp);

        //Check if allocaOp is an output in current genericOp
        for (auto user : sortedUsers) {
          if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {

            //auto genericOp = cast<linalg::GenericOp>(user);
            if(processedGenericOps.count(genericOp) > 0)
              continue;
            rewriter.setInsertionPointAfter(genericOp);

            SmallVector<Value, 4> newInputs;
            SmallVector<Value, 4> newOutputs;
            SmallVector<Type> resultTypes;
            //Create a new linalg.generic in Destination Style Passing format

            ArrayAttr indexingMaps = genericOp.getIndexingMaps();
            for(auto input : genericOp.getInputs()){
              newInputs.push_back(input == allocaOp ? currentTensor : input);
            }

            //ArrayRef<Type> resultTypes; 
            int newCurrentTensorIndex = -1;
            int index = 0;
            for(auto output : genericOp.getOutputs()){
              newOutputs.push_back(output == allocaOp ? currentTensor : output);
              resultTypes.push_back(output == allocaOp ? currentTensor.getType() : output.getType());
              if(output == allocaOp) {
                newCurrentTensorIndex = index;
              }
              index++;
            }

            StringAttr empty = StringAttr::get(genericOp.getContext());
            ArrayRef<Type> resultTypesRef(resultTypes);
            auto newGenericOp = rewriter.create<linalg::GenericOp>(genericOp.getLoc(), resultTypesRef, newInputs, newOutputs,
             genericOp.getIndexingMaps(), genericOp.getIteratorTypes(), empty, empty);

            Region &opRegion = newGenericOp.getRegion();
            rewriter.cloneRegionBefore(genericOp.getRegion(), newGenericOp.getRegion(), newGenericOp.getRegion().end());
          
            //Replace all uses of original generic op with the new one
            //int idxOldGeneric=0;
            //int idxNewGeneric=0;
            for (unsigned i = 0; i < genericOp->getNumResults(); ++i) {
              //if(i == newCurrentTensorIndex) {
              //  idxNewGeneric++;
              //}
              //genericOp->getResult(idxOldGeneric).replaceAllUsesWith(newGenericOp->getResult(idxNewGeneric));
              //idxOldGeneric++;
              //idxNewGeneric++;
              genericOp->getResult(i).replaceAllUsesWith(newGenericOp->getResult(i));
            }
          
            //Delete the original genericOp
            opsToDelete.push_back(genericOp.getOperation());
            if(newCurrentTensorIndex != -1)
              currentTensor = newGenericOp.getResult(newCurrentTensorIndex);
          
            processedGenericOps.insert(genericOp.getOperation());
          }
        }

        auto toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(
                                allocaOp.getLoc(),
                                allocaOp.getType(),
                                currentTensor);
        rewriter.create<memref::CopyOp>(allocaOp.getLoc(), toMemrefOp, allocaOp);
        //opsToDelete.push_back(allocaOp.getOperation());
        return WalkResult::advance();
    });
    for (Operation *op : opsToDelete) {
      op->erase();
    }
    opsToDelete.clear();

    return passResult;
  }
};

namespace {
struct LinalgDebufferize
    : public LinalgDebufferizeBase<LinalgDebufferize> {
  void runOnOperation() override;
};
} // namespace

void LinalgDebufferize::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.insert<LinalgDebufferization>(&getContext());
  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createLinalgDebufferizePass() {
  return std::make_unique<LinalgDebufferize>();
}
} // namespace polygeist
} // namespace mlir
