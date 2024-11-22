#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUEnableMoeLDSBypassPass
    : public TritonAMDGPUEnableMoeLDSBypassBase<
          TritonAMDGPUEnableMoeLDSBypassPass> {

public:
  TritonAMDGPUEnableMoeLDSBypassPass() = default;

  void runOnOperation() override {
    triton::enableMoeLDSBypass(true);
    // signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUEnableMoeLDSBypassPass() {
  return std::make_unique<TritonAMDGPUEnableMoeLDSBypassPass>();
}
