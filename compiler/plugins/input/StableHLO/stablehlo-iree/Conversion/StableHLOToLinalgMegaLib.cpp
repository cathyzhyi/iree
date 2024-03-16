// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO/CHLO dialects to Linalg dialect.

#include <algorithm>
#include <cstdint>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo-iree/Conversion/LegalizeToLinalgUtils.h"
#include "stablehlo-iree/Conversion/Passes.h"
#include "stablehlo-iree/Conversion/Rewriters.h"
#include "stablehlo-iree/Conversion/TypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOLINALGMEGALIB
#include "stablehlo-iree/Conversion/Passes.h.inc"

namespace {

static std::string getTypeToken(Type type) {
  if (type.isSignlessInteger())
    return ("i" + Twine(type.getIntOrFloatBitWidth())).str();
  else if (type.isa<mlir::FloatType>())
    return ("f" + Twine(type.getIntOrFloatBitWidth())).str();

  llvm_unreachable(
      "Type token should handle all types: memref, float and int type");
}

/// Helper function to convert a vector of int64_t into a vector of
/// `Value`s.
static SmallVector<Value> getAsValues(OpBuilder &b, Location loc,
                                      ArrayRef<int64_t> attrVec) {
  return llvm::to_vector<4>(
      llvm::map_range(attrVec, [&](int64_t value) -> Value {
        return b.create<arith::ConstantOp>(loc, b.getIndexAttr(value));
      }));
}

static SmallVector<Type> getAsTypes(OpBuilder &b, Location loc,
                                    ValueRange values) {
  return llvm::to_vector<4>(llvm::map_range(
      values, [&](Value val) -> Type { return val.getType(); }));
}

static std::pair<func::FuncOp, bool>
findOrCreateFuncOp(RewriterBase &rewriter, Operation *op, StringRef fnName,
                   TypeRange callArgumentTypes, TypeRange callReturnTypes) {
  FunctionType functionType =
      rewriter.getFunctionType(callArgumentTypes, callReturnTypes);
  bool newFunction = false;

  // Create a declaration for the function type.
  Location loc = op->getLoc();
  auto moduleOp = SymbolTable::getNearestSymbolTable(op);
  // Check for duplicates.
  auto fnDecl = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(moduleOp, fnName));
  if (!fnDecl) {
    newFunction = true;
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&moduleOp->getRegion(0).front());
    fnDecl = rewriter.create<func::FuncOp>(loc, fnName, functionType);
    SymbolTable::setSymbolVisibility(fnDecl, SymbolTable::Visibility::Public);
  }
  return std::make_pair(fnDecl, newFunction);
}

struct PadOpMegaLibConversion final
    : OpConversionPattern<mlir::stablehlo::PadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto edgePaddingLow = getAsValues(rewriter, loc, op.getEdgePaddingLow());
    auto edgePaddingHigh = getAsValues(rewriter, loc, op.getEdgePaddingHigh());
    Value paddingValue = rewriter.createOrFold<tensor::ExtractOp>(
        loc, adaptor.getPaddingValue());
    Value input = adaptor.getOperand();
    auto inputTy = llvm::cast<RankedTensorType>(input.getType());
    int64_t rank = inputTy.getRank();
    auto elemTy = inputTy.getElementType();
    // TODO: Negative edge padding
    SmallVector<Value, 4> args = {};
    args.push_back(adaptor.getOperand());
    args.append(edgePaddingLow);
    args.append(edgePaddingHigh);
    args.push_back(paddingValue);

    // TODO: change to dynamic dims
    // Type retTy = RankedTensorType::get({-1, -1}, elemTy);
    Type retTy = op.getType();
    auto argTys = getAsTypes(rewriter, loc, args);
    std::string funcName =
        ("mega_lib_pad_positive_" + Twine(rank) + "d_" + getTypeToken(elemTy))
            .str();
    auto [funcOp, insertFuncBody] =
        findOrCreateFuncOp(rewriter, op, funcName, argTys, retTy);

    if (insertFuncBody) {
      Region &funcRegion = funcOp.getRegion();
      Block &funcBlock = funcRegion.emplaceBlock();
      SmallVector<Location> locs(argTys.size(), loc);
      funcBlock.addArguments(argTys, locs);
      auto blockArgs = funcBlock.getArguments();
      auto input = blockArgs[0];
      SmallVector<OpFoldResult> edgePaddingLowFuncArgs;
      SmallVector<OpFoldResult> edgePaddingHighFuncArgs;
      for (int64_t i : llvm::seq<int64_t>(1, rank)) {
        edgePaddingLowFuncArgs.push_back(blockArgs[i]);
        edgePaddingHighFuncArgs.push_back(blockArgs[i + rank]);
      }
      int32_t paddingValueArgIndx = rank * 2 + 1;
      Value paddingValueFuncArg = blockArgs[paddingValueArgIndx];
      OpBuilder regionBuilder(funcRegion);

      //// TODO: interior padding
      //// If there is no interior padding lower to tensor.pad directly.
      auto padTensorOp = regionBuilder.create<tensor::PadOp>(
          loc, retTy, input, edgePaddingLowFuncArgs, edgePaddingHighFuncArgs,
          paddingValueFuncArg);
      regionBuilder.create<func::ReturnOp>(loc, padTensorOp.getResult());
    }

    auto callFun = rewriter.create<func::CallOp>(loc, funcOp, args);
    rewriter.replaceOp(op, callFun.getResult(0));
    return success();
  }
};

struct ConvertStableHloToLinalgMegaLib final
    : impl::ConvertStableHloToLinalgMegaLibBase<
          ConvertStableHloToLinalgMegaLib> {
  using ConvertStableHloToLinalgMegaLibBase::
      ConvertStableHloToLinalgMegaLibBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                    scf::SCFDialect, complex::ComplexDialect, math::MathDialect,
                    memref::MemRefDialect, shape::ShapeDialect>();
  }

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<
        bufferization::BufferizationDialect, arith::ArithDialect,
        complex::ComplexDialect, linalg::LinalgDialect, math::MathDialect,
        tensor::TensorDialect, sparse_tensor::SparseTensorDialect,
        scf::SCFDialect, shape::ShapeDialect, func::FuncDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    auto typeConverter = createStableHloToLinalgTypeConverter();
    ModuleOp module = getOperation();

    populateStableHloToLinalgMegaLibConversionPatterns(&ctx, *typeConverter,
                                                       &patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
//
void populateStableHloToLinalgMegaLibConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns) {

  // clang-format off
  patterns->add<
    PadOpMegaLibConversion
      >(typeConverter, context);
}
} // namespace mlir::iree_compiler::stablehlo
