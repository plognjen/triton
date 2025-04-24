#include "../TritonAMDGPUToLLVM/Utility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

template <typename T> unsigned getNumElements(const ArrayRef<T> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

struct ConcatOpConversion : public ConvertOpToLLVMPattern<amdgpu::ConcatOp> {
  using Base = ConvertOpToLLVMPattern<amdgpu::ConcatOp>;

  explicit ConcatOpConversion(LLVMTypeConverter &typeConverter,
                              PatternBenefit benefit = 1)
      : Base(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(amdgpu::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType resultType =
        cast<RankedTensorType>(op.getResult().getType());

    ArrayRef<int64_t> dstShape = resultType.getShape();
    Attribute dstEncoding = resultType.getEncoding();

    Value srcVal = op.getSources()[0];
    RankedTensorType srcType = cast<RankedTensorType>(srcVal.getType());
    ArrayRef<int64_t> srcShape = srcType.getShape();
    Attribute srcEncoding = srcType.getEncoding();

    auto linearEncodingSrc =
        triton::gpu::toLinearEncoding(srcEncoding, srcShape);
    auto linearEncodingDst =
        triton::gpu::toLinearEncoding(dstEncoding, dstShape);
    auto srcCTAOrder = linearEncodingSrc.getRepOrder();
    auto dstCTAOrder = linearEncodingDst.getRepOrder();

    auto rank = srcShape.size();
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(resultType);
    auto sources = adaptor.getSources();

    unsigned totalElems = ::getNumElements<long>(dstShape);
    unsigned elemsPerTile = ::getNumElements<unsigned>(shapePerCTATile);
    unsigned numCTATiles = totalElems / elemsPerTile;

    // Default order is minor-to-major.
    std::vector<unsigned> defaultOrder(rank);
    std::iota(defaultOrder.rbegin(), defaultOrder.rend(), 0);

    auto dstCTAShape =
        LLVM::AMD::multiDimDivision<long, unsigned>(dstShape, shapePerCTATile);
    auto srcCTAShape =
        LLVM::AMD::multiDimDivision<long, unsigned>(srcShape, shapePerCTATile);
    auto srcToDstShape =
        LLVM::AMD::multiDimDivision<long, long>(dstShape, srcShape);

    unsigned elemsPerThreadPerCTA =
        triton::gpu::getTotalElemsPerThread(srcType) /
        ::getNumElements<unsigned>(srcCTAShape);

    llvm::SmallVector<Value> resultVals;
    resultVals.reserve(totalElems);

    llvm::SmallVector<SmallVector<Value>> unpackedSources;
    unpackedSources.reserve(sources.size());

    for (int i = 0; i < sources.size(); i++) {
      Value currSrc = sources[i];
      unpackedSources.push_back(unpackLLElements(loc, currSrc, rewriter));
    }

    // Traverse CTA tiles in the result tensor
    for (int i = 0; i < numCTATiles; ++i) {
      auto currTileIdx = mlir::LLVM::delinearize(i, dstCTAShape, dstCTAOrder);
      // The n-dim destination tensor is built by arranging n-dim source tensors
      // into a destination tensor shape. Determine which source tensor contains
      // the current CTA tile.
      auto multiDimSrcIdx = LLVM::AMD::multiDimDivision<unsigned, unsigned>(
          currTileIdx, srcCTAShape);
      // Compute linear index of the current source tensor.
      // Concat operands are laid out in the destination tensor
      // in major‑to‑minor order.
      auto linearSrcIdx =
          mlir::LLVM::linearize(multiDimSrcIdx, srcToDstShape, defaultOrder);

      // After determining which source tensor the current CTA tile belongs to,
      // compute the index of this CTA tile within that source tensor,
      // considering the source tensor includes multiple CTA tiles.
      auto multiDimSrcCTAIdx = LLVM::AMD::multiDimReminder<unsigned, unsigned>(
          currTileIdx, srcCTAShape);
      auto linearSrcCTAIdx =
          mlir::LLVM::linearize(multiDimSrcCTAIdx, srcCTAShape, srcCTAOrder);
      auto unpackedElements = unpackedSources[linearSrcIdx];

      for (int j = 0; j < elemsPerThreadPerCTA; ++j)
        resultVals.push_back(
            unpackedElements[linearSrcCTAIdx * elemsPerThreadPerCTA + j]);
    }

    Value packedResult = packLLElements(loc, this->getTypeConverter(),
                                        resultVals, rewriter, resultType);

    rewriter.replaceOp(op, packedResult);
    return success();
  }
};
} // namespace

namespace mlir::triton::AMD {
void populateConcatOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns,
                                    mlir::PatternBenefit benefit) {
  patterns.add<ConcatOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
