#include "third_party/amd/include/TritonAMDGPUToLLVM/MembarUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "third_party/amd/include/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
namespace {

// Returns true if one of the operands is a LocalLoad synced via AsyncWait.
bool filterAsyncLocalLoadsDeppendencies(Operation *op1, Operation *op2) {
  auto isAsyncLoad = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                     triton::amdgpu::BufferLoadToLocalOp>(op);
  };
  auto isLocalLoadWithAsyncWaitToken = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    if (!localLoad)
      return false;
    Value token = localLoad.getToken();
    if (!token || !comesFromAsyncWait(token))
      return false;
    return true;
  };

  // TODO (alex): WA because we get a barrier between sliced AsyncCopies because
  // they write to the same LDS allocation
  if (isAsyncLoad(op1) && isAsyncLoad(op2)) {
    return true;
  }

  // Early return if neither or both operands are an AsyncLoad
  if (isAsyncLoad(op1) == isAsyncLoad(op2)) {
    return false;
  }

  return isLocalLoadWithAsyncWaitToken(op1) ||
         isLocalLoadWithAsyncWaitToken(op2);
};
} // namespace

bool membarFilter(Operation *op1, Operation *op2) {
  return filterAsyncLocalLoadsDeppendencies(op1, op2);
}
} // namespace mlir::triton::AMD
