// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/InputConversion/TMTensor/PassDetail.h"
#include "iree/compiler/InputConversion/TMTensor/Passes.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir::torch;

namespace mlir {
namespace iree_compiler {
namespace TMTensor {

namespace {

template <typename SrcOpTy, typename TargetOpTy>
struct TMTensorOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    OperationState state(srcOp->getLoc(), TargetOpTy::getOperationName(),
                         srcOp->getOperands(), srcOp->getResultTypes(),
                         srcOp->getAttrs(), srcOp->getSuccessors());
    for (Region &r : srcOp->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
    }
    Operation *newOp = rewriter.createOperation(state);
    rewriter.replaceOp(srcOp, newOp->getResults());
    return success();
  }
};
}  // namespace

namespace {

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertTMTensorToLinalgExtPass
    : public ConvertTMTensorToLinalgExtBase<ConvertTMTensorToLinalgExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

#define INSERT_TMTENSOR_CONVERSION_PATTERN(Op)                               \
  patterns.add<                                                              \
      TMTensorOpConversion<mlir::torch::TMTensor::Op, IREE::LinalgExt::Op>>( \
      typeConverter, context);

    INSERT_TMTENSOR_CONVERSION_PATTERN(YieldOp);
    INSERT_TMTENSOR_CONVERSION_PATTERN(ScatterOp);
    INSERT_TMTENSOR_CONVERSION_PATTERN(ScanOp);

#undef INSERT_TMTENSOR_CONVERSION_PATTERN

    target.addIllegalDialect<mlir::torch::TMTensor::TMTensorDialect>();
    target.addLegalDialect<IREE::LinalgExt::IREELinalgExtDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertTMTensorToLinalgExtPass() {
  return std::make_unique<ConvertTMTensorToLinalgExtPass>();
}

}  // namespace TMTensor
}  // namespace iree_compiler
}  // namespace mlir
