#ifndef TRITON_GPU_DIALECT_INTERFACES_H
#define TRITON_GPU_DIALECT_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "triton/Dialect/TritonGPU/IR/CGAEncodingAttr.h"

// clang-format off
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

// Free functions called by LinearEncodingTrait default implementations.
// Declared here so they are visible to the generated Trait code in
// AttrInterfaces.h.inc.
namespace mlir::triton::gpu::linear_encoding_impl {

SmallVector<unsigned> basesPerDim(const LinearLayout &ll, StringAttr dimName,
                                  bool skipBroadcast);
SmallVector<unsigned> orderPerDim(const LinearLayout &ll, StringAttr dimName,
                                  ArrayRef<unsigned> defaultOrder);
SmallVector<unsigned> getContig(const LinearLayout &ll, MLIRContext *ctx,
                                const char *inDim,
                                SmallVector<unsigned> lowerContig,
                                ArrayRef<unsigned> order);
SmallVector<unsigned> getSizePerThread(const LinearLayout &ll, MLIRContext *ctx,
                                       ArrayRef<unsigned> cgaSplitNum);
CGAEncodingAttr getCGALayout(const LinearLayout &ll, MLIRContext *ctx);
LinearLayout toLinearLayout(const LinearLayout &ll,
                            ArrayRef<unsigned> repOrder,
                            ArrayRef<int64_t> shape);
SmallVector<unsigned> getElemsPerThread(const LinearLayout &ll,
                                        MLIRContext *ctx,
                                        ArrayRef<unsigned> repOrder,
                                        ArrayRef<int64_t> shape);
unsigned getTotalElemsPerThread(const LinearLayout &ll, MLIRContext *ctx,
                                ArrayRef<unsigned> repOrder,
                                ArrayRef<int64_t> shape);

} // namespace mlir::triton::gpu::linear_encoding_impl

#include "triton/Dialect/TritonGPU/IR/AttrInterfaces.h.inc"
#include "triton/Dialect/TritonGPU/IR/OpInterfaces.h.inc"
// clang-format on

#endif // TRITON_GPU_DIALECT_INTERFACES_H
