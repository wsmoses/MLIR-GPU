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



//module @harris_score_with_gradient_extra_kernel {
//    memref.global "private" @_ZL8coeffs_1 : memref<5x5xi32> = dense<1>
//    memref.global "private" @_ZL8coeffs_y : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
//    memref.global "private" @_ZL8coeffs_x : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
//    memref.global @score : memref<512x512xi32> = uninitialized
//    func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
//      %c4_i32 = arith.constant 4 : i32
//      %c0_i32 = arith.constant 0 : i32
//      %alloca = memref.alloca() : memref<512x512xi32>
//      %alloca_0 = memref.alloca() : memref<512x512xi32>
//      %alloca_1 = memref.alloca() : memref<512x512xi32>
//      %alloca_2 = memref.alloca() : memref<516x516xi32>
//      %alloca_3 = memref.alloca() : memref<516x516xi32>
//      %alloca_4 = memref.alloca() : memref<518x518xi32>
//      %0 = memref.get_global @_ZL8coeffs_x : memref<3x3xi32>
//      %1 = memref.get_global @_ZL8coeffs_y : memref<3x3xi32>
//      %2 = memref.get_global @_ZL8coeffs_1 : memref<5x5xi32>
//      // 2nd variant
//      // %0 = memref.alloca() : memref<3x3xi32>
//      // %1 = memref.alloca() : memref<3x3xi32>
//      // %2 = memref.alloca() : memref<5x5xi32>
//      linalg.generic {indexing_maps = [#map17, #map18, #map18, #map19, #map19], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%alloca_4, %0, %1 : memref<518x518xi32>, memref<3x3xi32>, memref<3x3xi32>) outs(%alloca_2, %alloca_3 : memref<516x516xi32>, memref<516x516xi32>) {
//      ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32, %out_7: i32):
//        %4 = arith.muli %in, %in_5 : i32
//        %5 = arith.addi %out_7, %4 : i32
//        %6 = arith.muli %in, %in_6 : i32
//        %7 = arith.addi %out, %6 : i32
//        linalg.yield %7, %5 : i32, i32
//      }
//      linalg.generic {indexing_maps = [#map17, #map17, #map18, #map19, #map19, #map19], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%alloca_3, %alloca_2, %2 : memref<516x516xi32>, memref<516x516xi32>, memref<5x5xi32>) outs(%alloca, %alloca_0, %alloca_1 : memref<512x512xi32>, memref<512x512xi32>, memref<512x512xi32>) {
//      ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32, %out_7: i32, %out_8: i32):
//        %4 = arith.muli %in, %in : i32
//        %5 = arith.muli %4, %in_6 : i32
//        %6 = arith.addi %out_8, %5 : i32
//        %7 = arith.muli %in_5, %in_5 : i32
//        %8 = arith.muli %7, %in_6 : i32
//        %9 = arith.addi %out_7, %8 : i32
//        %10 = arith.muli %in, %in_5 : i32
//        %11 = arith.muli %10, %in_6 : i32
//        %12 = arith.addi %out, %11 : i32
//        linalg.yield %12, %9, %6 : i32, i32, i32
//      }
//      %3 = memref.get_global @score : memref<512x512xi32>
//      linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel"]} ins(%alloca_1, %alloca_0, %alloca : memref<512x512xi32>, memref<512x512xi32>, memref<512x512xi32>) outs(%3 : memref<512x512xi32>) {
//      ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32):
//        %4 = arith.muli %in, %in_5 : i32
//        %5 = arith.muli %in_6, %in_6 : i32
//        %6 = arith.subi %4, %5 : i32
//        %7 = arith.addi %in, %in_5 : i32
//        %8 = arith.muli %7, %c4_i32 : i32
//        %9 = arith.muli %8, %7 : i32
//        %10 = arith.subi %6, %9 : i32
//        linalg.yield %10 : i32
//      }
//      return %c0_i32 : i32
//    }
//  }
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

        //Check if allocaOp is an output in current genericOp
        for (auto user : allocaOp->getUsers()) {
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
              resultTypes.push_back(currentTensor.getType());
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
            int idxOldGeneric=0;
            int idxNewGeneric=0;
            for (unsigned i = 0; i < genericOp->getNumResults(); ++i) {
              if(i == newCurrentTensorIndex) {
                idxNewGeneric++;
              }
              genericOp->getResult(i).replaceAllUsesWith(newGenericOp->getResult(i));
              idxOldGeneric++;
              idxNewGeneric++;
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
