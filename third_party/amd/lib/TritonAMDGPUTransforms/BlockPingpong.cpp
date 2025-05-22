#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#define DEBUG_TYPE "tritonamdgpu-block-pingpong"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {

// This pass transforms a for-loop calculating a GEMM. Main purpose of the
// transform is improve the efficiency of the GPU dot instruction (mfma)
// by interleaving the execution of two warps on each SIMD. Especially it groups
// instructions into Dot and Memory clusters so they can efficiently run in
// parallel. Also this pass inserts `rocdl.s.setprio` operation and
// `amdgpu.cond_barrier` to run two parallel warps in synchronization.
// This scheduling doesn't help improving the memory latency itself but it
// relies on software-pipelining to hide the global latency. Likely to improve
// the performance of compute-bound cases.
class Pingponger {
  scf::ForOp forOp;
  SmallVector<tt::LoadOp> gLoadOps;
  SmallVector<ttg::LocalLoadOp> lLoadOps;
  SmallVector<ttg::LocalStoreOp> lStoreOps;
  SmallVector<ttg::AsyncCopyGlobalToLocalOp> asyncCopyOps;
  SmallVector<ttg::AsyncWaitOp> asyncWaitOps;
  SmallVector<ttg::AsyncCommitGroupOp> asyncCommitOps;
  SmallVector<tt::DotOp> dotOps;
  SmallVector<tt::DotScaledOp> dotSOps;
  SmallVector<SmallVector<Operation *>> subViewOps;
  SmallVector<SmallVector<Operation *>> loadSliceOps;
  SmallVector<Operation *> dotSliceOps;
  SmallVector<Value> constOffsets;
  Operation *lastInsertedOp;

  // rocdl.s.setprio will be mapped to `s_setprio` instruction which set the
  // priority of the warp within a SIMD, determines which warp to occupy the
  // instruction unit when they compete on the same instruction.
  // We use this instruction in the pingpong scheduling to prevent warps from
  // entering into the dot cluster while the other warp is still busy in the dot
  // cluster. Otherwise pingpong pattern can be broken and performance drops.
  // Currently pingpong only handles two warps, we only need 0/1 priorities.
  int lowPriority = 0;
  int highPriority = 1;
  int32_t kWidth;
  int32_t numWarps;
  int32_t numStages;
  bool useAsyncCopy;

public:
  Pingponger(scf::ForOp forOp, int32_t numWarps, int32_t numStages,
             bool useAsyncCopy)
      : forOp(forOp), numWarps(numWarps), numStages(numStages),
        useAsyncCopy(useAsyncCopy) {}
  void getDotPingponged();

private:
  void genOffsetConstants(Location loc, OpBuilder &builder, unsigned numSlices,
                          int64_t sliceWidth);
  LogicalResult genLocalSlice(OpBuilder &builder, Value v,
                              Attribute dotEncoding, unsigned opIdx,
                              unsigned numSlices, int64_t sliceWidth);
  LogicalResult genLocalSliceScales(OpBuilder &builder, Value v,
                                    Attribute dotEncoding, unsigned opIdx,
                                    unsigned numSlices, int64_t sliceWidth);
  LogicalResult sliceDot(OpBuilder &builder, Location loc, tt::DotOp op,
                         unsigned numSlices);
  LogicalResult sliceDotScaled(OpBuilder &builder, Location loc,
                               tt::DotScaledOp op, unsigned numSlices);
  void transformOnePPClusters(OpBuilder &builder, Location loc);
  LogicalResult transformFourPPClusters(OpBuilder &builder, Location loc);
  LogicalResult transformTwoPPClusters(OpBuilder &builder, Location loc);
  LogicalResult transformFAv3(OpBuilder &builder, Location loc);
  LogicalResult transformFP4(OpBuilder &builder, Location loc);
  LogicalResult transformFP4s(OpBuilder &builder, Location loc);
  LogicalResult transformFP4mn(OpBuilder &builder, Location loc);
  void addAsymmetricSyncToLoop(OpBuilder &builder, Location loc);
  void updateOpInsertion(Operation *Op);
  void appendOp(Operation *Op);
  void prependOp(Operation *Op, bool moveBackwards);
  void moveOpAndPredecessorsUpSameBlock(Operation *Op);
  void appendSlicedLoadAB(int slice);
  SmallVector<Operation *> genClusterBarrier(OpBuilder &builder, Location loc);
  LogicalResult genScaleSlice(OpBuilder &builder, Value v, unsigned numSlices,
                              int sliceDim);
  SmallVector<Value> genSplitAsyncCopy(OpBuilder &builder, Value v,
                                       unsigned numSlices, unsigned sliceDim);
  LogicalResult genOperandSlice(OpBuilder &builder, Value v, unsigned numSlices,
                                int sliceDim);
  void appendClusterBarrier(OpBuilder &builder, Location loc);
  void prependClusterBarrier(OpBuilder &builder, Location loc);
  void appendOpWithPrio(OpBuilder &builder, Operation *Op, Location loc);
  bool isPersistentGemm(size_t num_dots);
  template <typename T>
  size_t countIfMemoryOps(scf::IfOp ifOp, bool assumeNotTaken);
  template <typename T>
  size_t estimateNonDotMemoryImpact(T *start, T *end, bool assumeNotTaken);
  void determineDotMemoryOps(tt::DotOp dotOp,
                             DenseSet<tt::LoadOp> &dotGlobalLoads,
                             DenseSet<ttg::LocalLoadOp> &dotLocalLoads,
                             DenseSet<ttg::LocalStoreOp> &dotLocalStores);
  LogicalResult pruneDotMemoryOps(DenseSet<tt::LoadOp> &dotGlobalLoads,
                                  DenseSet<ttg::LocalLoadOp> &dotLocalLoads,
                                  DenseSet<ttg::LocalStoreOp> &dotLocalStores,
                                  bool assumeNotTaken);
  void determineDotAsyncMemoryOps(
      tt::DotOp dotOp,
      DenseSet<ttg::AsyncCopyGlobalToLocalOp> &dotAsyncGlobalLoads,
      DenseSet<ttg::LocalLoadOp> &dotLocalLoads,
      DenseSet<ttg::AsyncCommitGroupOp> &dotAsyncCommitGroups,
      DenseSet<ttg::AsyncWaitOp> &dotAsyncWaits);
  LogicalResult pruneDotAsyncMemoryOps(
      DenseSet<ttg::AsyncCopyGlobalToLocalOp> &dotGlobalLoads,
      DenseSet<ttg::LocalLoadOp> &dotLocalLoads,
      DenseSet<ttg::AsyncCommitGroupOp> &dotAsyncCommitGroups,
      DenseSet<ttg::AsyncWaitOp> &dotAsyncWaits, bool assumeNotTaken);
  template <typename T>
  void findClosestPredOps(Value v, DenseSet<T> &matchingOps);

  LogicalResult genLocalSliceHelper(OpBuilder &builder, Value v, unsigned opIdx,
                                    unsigned numSlices, int64_t sliceWidth,
                                    RankedTensorType tensorType);
};

void Pingponger::updateOpInsertion(Operation *op) { lastInsertedOp = op; }
void Pingponger::appendOp(Operation *op) {
  assert(lastInsertedOp != nullptr);
  op->moveAfter(lastInsertedOp);
  lastInsertedOp = op;
}
void Pingponger::prependOp(Operation *op, bool moveBackwards) {
  assert(lastInsertedOp != nullptr);
  op->moveBefore(lastInsertedOp);
  if (moveBackwards)
    lastInsertedOp = op;
}

// Move the given operations and any predecessors upon which it depends
// up in the block to the last inserted operation. This does not move
// operations that reaches the last inserted operation or
// are not in the same block. The exception is op, which is always moved
// to the new location (can move down or up).
void Pingponger::moveOpAndPredecessorsUpSameBlock(Operation *op) {
  assert(lastInsertedOp != nullptr);
  // TODO: Enable moving ops across blocks
  assert(op->getBlock() == lastInsertedOp->getBlock());
  Operation *checkedOp = lastInsertedOp;
  // Check if we are moving the op up, if so we may need to
  // move additional ops up to maintain correctness.
  if (lastInsertedOp->isBeforeInBlock(op)) {
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = [&checkedOp](Operation *op) {
      return op->getBlock() == checkedOp->getBlock() &&
             checkedOp->isBeforeInBlock(op);
    };
    getBackwardSlice(op, &backwardSlice, opt);
    for (auto predOp : backwardSlice)
      appendOp(predOp);
    appendOp(op);
  } else {
    auto hasUnsafeUser = [&checkedOp](auto &&user) {
      return user != checkedOp && user->getBlock() == checkedOp->getBlock() &&
             user->isBeforeInBlock(checkedOp);
    };
    if (std::any_of(op->user_begin(), op->user_end(), hasUnsafeUser))
      LDBG("Unable to move operation "
           << op << " due to use before intended move location");
    else
      appendOp(op);
  }
}
void Pingponger::appendSlicedLoadAB(int slice) {
  appendOp(subViewOps[0][slice]);
  appendOp(loadSliceOps[0][slice]);
  appendOp(subViewOps[1][slice]);
  appendOp(loadSliceOps[1][slice]);
}
// Asymmetrically synchronized loop in the pingpong scheduling synchronizes all
// the warps at the end of each instruction cluster. Since cond_barrier
// triggered a barrier for only half of the warps in a block, at the point
// this clusterBarrier is called, half warps are at dot cluster and the others
// are at the memory cluster.
// Also, SchedBarrier with `0` is set here to tell compiler backend not to
// reorder any instruction across this point.
SmallVector<Operation *> Pingponger::genClusterBarrier(OpBuilder &builder,
                                                       Location loc) {
  //  MembarAnalysis can recognize gpu::BarrierOp and skip inserting additional
  auto barrierOp = builder.create<gpu::BarrierOp>(loc);
  auto schedBarrierOp = builder.create<ROCDL::SchedBarrier>(loc, 0);
  return {barrierOp, schedBarrierOp};
}
void Pingponger::appendClusterBarrier(OpBuilder &builder, Location loc) {
  for (auto &&op : genClusterBarrier(builder, loc))
    appendOp(op);
}
void Pingponger::prependClusterBarrier(OpBuilder &builder, Location loc) {
  for (auto &&op : genClusterBarrier(builder, loc))
    prependOp(op, false);
}
void Pingponger::appendOpWithPrio(OpBuilder &builder, Operation *op,
                                  Location loc) {
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, highPriority));
  appendOp(op);
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, lowPriority));
}

// Determine if the given loop matches the basic pattern of a persistent GEMM.
// Here we define a persistent GEMM as containing a single dot product, and two
// if statements inside the body of the loop. While canonically these should be
// var == 0 and var == other_var - 1, we approximate this check to just check
// for a comparison equality. This will miss legal variant like >= var and we
// can adjust this with example kernels that fail.
//
// Note: That while ideally we would check that these are the same variable
// and that they change per loop iteration, the persistent GEMM cannot depend
// directly on the loop bounds, we will avoid matching an exact pattern which
// may be quite flexible in general.
bool Pingponger::isPersistentGemm(size_t num_dots) {
  if (num_dots != 1)
    return false;
  bool seenIfSection = false;
  bool seenDot = false;
  for (auto &op : *forOp.getBody()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (seenIfSection) {
        // Violate our two if statement assumption.
        return false;
      }
      auto cond = ifOp.getCondition().getDefiningOp();
      if (!cond) {
        return false;
      }
      bool matchesPattern = false;
      if (auto cmpIOp = dyn_cast<arith::CmpIOp>(cond)) {
        matchesPattern =
            cmpIOp.getPredicate() == mlir::arith::CmpIPredicate::eq;
      }
      if (!matchesPattern) {
        return false;
      }
      seenIfSection = true;
    } else if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
      if (seenDot || !seenIfSection) {
        // Violate structure of the persistent GEMM
        // assumption.
        return false;
      }
      seenDot = true;
      // Reset the if section flag.
      seenIfSection = false;
    }
  }
  return seenIfSection && seenDot;
}

// Find all of the "closest" operations that are of a given type T
// in the same basic block. Here "closest" means along any path P,
// the first operation of type T that is encountered when traversing
// P from the given value v. This also includes "later" operations
// for block arguments. Note: That we find all T for every path P.
template <typename T>
void Pingponger::findClosestPredOps(Value v, DenseSet<T> &matchingOps) {
  // Create a cache so we can traverse across block arguments.
  DenseSet<Operation *> visitedOps;
  std::function<void(Value)> impl;
  impl = [&matchingOps, &visitedOps, &impl](Value v) {
    // If we encounter a block argument we only look at the terminators of the
    // current block
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      auto operandNumber = blockArg.getArgNumber();
      auto block = blockArg.getOwner();
      if (auto yield = dyn_cast<scf::YieldOp>(block->getTerminator())) {
        auto parentOp = block->getParentOp();
        // Skip the induction variables to find the yield position
        if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
          if (operandNumber < forOp.getNumInductionVars())
            return;
          operandNumber -= forOp.getNumInductionVars();
        }
        impl(yield->getOperand(operandNumber));
      }
    } else {
      auto definingOp = v.getDefiningOp();
      if (!definingOp)
        return;
      else if (visitedOps.contains(definingOp))
        return;
      visitedOps.insert(definingOp);
      if (auto matchOp = dyn_cast<T>(definingOp))
        matchingOps.insert(matchOp);
      else
        for (auto predValue : definingOp->getOperands())
          impl(predValue);
    }
  };
  impl(v);
}

// Determine the number of memory operations of type T that are expected
// to execute each iteration of the outermost for loop for the ifOp.
template <typename T>
size_t Pingponger::countIfMemoryOps(scf::IfOp ifOp, bool assumeNotTaken) {
  // Don't do a nested traversal as we are only estimating the "same level"
  auto thenOps = ifOp.thenBlock()->getOps<T>();
  size_t thenCount = std::distance(thenOps.begin(), thenOps.end());
  size_t elseCount = 0;
  if (ifOp.elseBlock()) {
    auto elseOps = ifOp.elseBlock()->getOps<T>();
    elseCount = std::distance(elseOps.begin(), elseOps.end());
  }
  // Estimate the worst case unless we have assumeNotTaken == true.
  return assumeNotTaken ? elseCount : std::max(thenCount, elseCount);
}

// Estimate the expected number of memory operations of type T
// rounded to an integer. This is used to determine any possible
// influence on cluster setup.
template <typename T>
size_t Pingponger::estimateNonDotMemoryImpact(T *start, T *end,
                                              bool assumeNotTaken) {
  DenseSet<Operation *> visitedParents;
  size_t count = 0;
  for (auto it = start; it != end; it++) {
    auto parent = (*it)->getParentOp();
    if (parent == nullptr)
      continue;
    if (parent == forOp)
      count += 1;
    else {
      if (visitedParents.contains(parent))
        continue;
      visitedParents.insert(parent);
      if (auto ifOp = dyn_cast<scf::IfOp>(parent))
        count += countIfMemoryOps<T>(ifOp, assumeNotTaken);
      else {
        // Default to counting every memory access as a
        // single access.
        count += 1;
      }
    }
  }
  return count;
}

// Populate the dotGlobalLoads, dotLocalLoads, and dotLocalStores set with
// any loads that are generated by the current dot product. This occurs in
// steps to:
// 1. Determine which loads are generated by the dot product via getA()
//    and getB().
// 2. Determine which local stores are used to populate the inputs to
//    the local loads.
// 3. Determine which global loads are used to populate the inputs to
//    the local stores.
// Note: This function currently depends on num_stages=2, which is a
// precondition for the pingpong scheduling.
void Pingponger::determineDotMemoryOps(
    tt::DotOp dotOp, DenseSet<tt::LoadOp> &dotGlobalLoads,
    DenseSet<ttg::LocalLoadOp> &dotLocalLoads,
    DenseSet<ttg::LocalStoreOp> &dotLocalStores) {
  // Find the locals loads used to compute the dot inputs. These
  // must come before the dot op.
  findClosestPredOps<ttg::LocalLoadOp>(dotOp.getA(), dotLocalLoads);
  findClosestPredOps<ttg::LocalLoadOp>(dotOp.getB(), dotLocalLoads);

  // Determine the local stores from the local loads.
  // With pipelining we expect this to be a single local
  // store within the loop based on a block argument after routing through
  // a ttg.MemDescSubviewOp.
  DenseSet<ttg::MemDescSubviewOp> subviews;
  for (auto &&localLoad : dotLocalLoads)
    findClosestPredOps<ttg::MemDescSubviewOp>(localLoad.getSrc(), subviews);

  for (auto &&subview : subviews)
    for (auto &&user : subview->getUsers())
      if (auto localStore = dyn_cast<ttg::LocalStoreOp>(user))
        dotLocalStores.insert(localStore);

  // Determine the global loads from the local stores.
  // We expect this to just be a global load
  // within the loop.
  for (auto &&localStore : dotLocalStores)
    findClosestPredOps<tt::LoadOp>(localStore.getSrc(), dotGlobalLoads);
}

// Populate the dotAsyncGlobalLoads, dotLocalLoads, dotAsyncCommitGroups, and
// dotAsyncWaits set with any loads that are generated by the current dot
// product. This occurs in steps to:
// 1. Determine which loads are generated by the dot product via getA()
//    and getB().
// 2. Determine which asyncCopyGlobalToLcals are used to populate the
// inputs to the local loads.
// 3. Determine which async commit are using asyncCopyGlobalToLcals.
// 4. Determine which async waits are consuming async commits
// Note: This function currently depends on num_stages=2, which is a
// precondition for the pingpong scheduling.
void Pingponger::determineDotAsyncMemoryOps(
    tt::DotOp dotOp,
    DenseSet<ttg::AsyncCopyGlobalToLocalOp> &dotAsyncGlobalLoads,
    DenseSet<ttg::LocalLoadOp> &dotLocalLoads,
    DenseSet<ttg::AsyncCommitGroupOp> &dotAsyncCommitGroups,
    DenseSet<ttg::AsyncWaitOp> &dotAsyncWaits) {
  // Find the locals loads used to compute the dot inputs. These
  // must come before the dot op.
  findClosestPredOps<ttg::LocalLoadOp>(dotOp.getA(), dotLocalLoads);
  findClosestPredOps<ttg::LocalLoadOp>(dotOp.getB(), dotLocalLoads);

  // Determine the local stores from the local loads.
  // With pipelining we expect this to be a single local
  // store within the loop based on a block argument after routing through
  // a ttg.MemDescSubviewOp.
  DenseSet<ttg::MemDescSubviewOp> subviews;
  for (auto &&localLoad : dotLocalLoads)
    findClosestPredOps<ttg::MemDescSubviewOp>(localLoad.getSrc(), subviews);

  for (auto &&subview : subviews) {
    for (auto &&user : subview->getUsers()) {
      if (auto globalLoad = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(user)) {
        if (!globalLoad->hasOneUse())
          continue;
        auto asyncCommitGroup =
            dyn_cast<ttg::AsyncCommitGroupOp>(*globalLoad->getUsers().begin());
        if (!asyncCommitGroup)
          continue;

        dotAsyncGlobalLoads.insert(globalLoad);
        dotAsyncCommitGroups.insert(asyncCommitGroup);
      }
    }
  }

  // Looks for AsyncWaitOp, which after StreamPipeliner should be
  // located/consumed by the iter arg which represent the AsyncCommits.
  for (auto &&asyncCommitGroup : dotAsyncCommitGroups) {
    if (!asyncCommitGroup->hasOneUse())
      return;
    auto asyncWaitOp =
        dyn_cast<ttg::AsyncWaitOp>(*asyncCommitGroup->getUsers().begin());
    if (!asyncWaitOp)
      return;
    dotAsyncWaits.insert(asyncWaitOp);
  }
}

LogicalResult
Pingponger::pruneDotMemoryOps(DenseSet<tt::LoadOp> &dotGlobalLoads,
                              DenseSet<ttg::LocalLoadOp> &dotLocalLoads,
                              DenseSet<ttg::LocalStoreOp> &dotLocalStores,
                              bool assumeNotTaken) {
  // Prune Memory operations that may be moved to only those involved in dot
  // computation. To understand the "cluster assumptions" we also estimate
  // the impact of any additional loads/stores.
  auto gLoadIt = std::stable_partition(
      gLoadOps.begin(), gLoadOps.end(),
      [&dotGlobalLoads](tt::LoadOp op) { return dotGlobalLoads.contains(op); });
  auto lLoadIt = std::stable_partition(lLoadOps.begin(), lLoadOps.end(),
                                       [&dotLocalLoads](ttg::LocalLoadOp op) {
                                         return dotLocalLoads.contains(op);
                                       });
  auto lStoreIt =
      std::stable_partition(lStoreOps.begin(), lStoreOps.end(),
                            [&dotLocalStores](ttg::LocalStoreOp op) {
                              return dotLocalStores.contains(op);
                            });

  if (estimateNonDotMemoryImpact<tt::LoadOp>(gLoadIt, gLoadOps.end(),
                                             assumeNotTaken) != 0) {
    std::stringstream message;
    message << "Unable to match ping pong scheduling pattern. Details: "
            << "Non-dot global loads found in non-persistent GEMM";
    LDBG(message.str());
    return failure();
  }
  if (estimateNonDotMemoryImpact<ttg::LocalLoadOp>(lLoadIt, lLoadOps.end(),
                                                   assumeNotTaken) != 0) {
    std::stringstream message;
    message << "Unable to match ping pong scheduling pattern. Details: "
            << "Non-dot local loads found in non-persistent GEMM";
    LDBG(message.str());
    return failure();
  }
  if (estimateNonDotMemoryImpact<ttg::LocalStoreOp>(lStoreIt, lStoreOps.end(),
                                                    assumeNotTaken) != 0) {
    std::stringstream message;
    message << "Unable to match ping pong scheduling pattern. Details: "
            << "Non-dot local stores found in non-persistent GEMM";
    LDBG(message.str());
    return failure();
  }

  // Remove non-dot memory operations.
  gLoadOps.erase(gLoadIt, gLoadOps.end());
  lLoadOps.erase(lLoadIt, lLoadOps.end());
  lStoreOps.erase(lStoreIt, lStoreOps.end());
  // All PingPong Scheduler assumes there are 2 movable global loads and 2
  // movable local loads.
  if (gLoadOps.size() != 2 || lLoadOps.size() != 2) {
    std::stringstream message;
    message << "Unable to match ping pong slicing pattern. Details: "
            << gLoadOps.size() << " global loads in dot computation, "
            << lLoadOps.size() << " local loads in dot computation";
    LDBG(message.str());
    return failure();
  }
  return success();
}

LogicalResult Pingponger::pruneDotAsyncMemoryOps(
    DenseSet<ttg::AsyncCopyGlobalToLocalOp> &dotGlobalLoads,
    DenseSet<ttg::LocalLoadOp> &dotLocalLoads,
    DenseSet<ttg::AsyncCommitGroupOp> &dotAsyncCommitGroups,
    DenseSet<ttg::AsyncWaitOp> &dotAsyncWaits, bool assumeNotTaken) {
  // Prune Memory operations that may be moved to only those involved in dot
  // computation. To understand the "cluster assumptions" we also estimate
  // the impact of any additional loads/stores.
  auto asyncCopyIt = std::stable_partition(
      asyncCopyOps.begin(), asyncCopyOps.end(),
      [&dotGlobalLoads](ttg::AsyncCopyGlobalToLocalOp op) {
        return dotGlobalLoads.contains(op);
      });
  auto lLoadIt = std::stable_partition(lLoadOps.begin(), lLoadOps.end(),
                                       [&dotLocalLoads](ttg::LocalLoadOp op) {
                                         return dotLocalLoads.contains(op);
                                       });
  auto asyncCommitIt = std::stable_partition(
      asyncCommitOps.begin(), asyncCommitOps.end(),
      [&dotAsyncCommitGroups](ttg::AsyncCommitGroupOp op) {
        return dotAsyncCommitGroups.contains(op);
      });
  auto asyncWaitIt =
      std::stable_partition(asyncWaitOps.begin(), asyncWaitOps.end(),
                            [&dotAsyncWaits](ttg::AsyncWaitOp op) {
                              return dotAsyncWaits.contains(op);
                            });

  if (estimateNonDotMemoryImpact<ttg::AsyncCopyGlobalToLocalOp>(
          asyncCopyIt, asyncCopyOps.end(), assumeNotTaken) != 0) {
    std::stringstream message;
    message << "Unable to match ping pong scheduling pattern. Details: "
            << "Non-dot global loads found in non-persistent GEMM";
    LDBG(message.str());
    return failure();
  }
  if (estimateNonDotMemoryImpact<ttg::LocalLoadOp>(lLoadIt, lLoadOps.end(),
                                                   assumeNotTaken) != 0) {
    std::stringstream message;
    message << "Unable to match ping pong scheduling pattern. Details: "
            << "Non-dot local loads found in non-persistent GEMM";
    LDBG(message.str());
    return failure();
  }
  if (estimateNonDotMemoryImpact<ttg::AsyncCommitGroupOp>(
          asyncCommitIt, asyncCommitOps.end(), assumeNotTaken) != 0) {
    std::stringstream message;
    message << "Unable to match ping pong scheduling pattern. Details: "
            << "Non-dot local stores found in non-persistent GEMM";
    LDBG(message.str());
    return failure();
  }
  if (estimateNonDotMemoryImpact<ttg::AsyncWaitOp>(
          asyncWaitIt, asyncWaitOps.end(), assumeNotTaken) != 0) {
    std::stringstream message;
    message << "Unable to match ping pong scheduling pattern. Details: "
            << "Non-dot local stores found in non-persistent GEMM";
    LDBG(message.str());
    return failure();
  }

  // Remove non-dot memory operations.
  asyncCopyOps.erase(asyncCopyIt, asyncCopyOps.end());
  lLoadOps.erase(lLoadIt, lLoadOps.end());
  asyncCommitOps.erase(asyncCommitIt, asyncCommitOps.end());
  asyncWaitOps.erase(asyncWaitIt, asyncWaitOps.end());
  // All PingPong Scheduler assumes there are 2 movable global loads and 2
  // movable local loads.
  if (asyncCopyOps.size() != 2 || lLoadOps.size() != 2 ||
      asyncWaitOps.size() != 2) {
    std::stringstream message;
    message << "Unable to match ping pong slicing pattern. Details: "
            << asyncCopyOps.size() << " global loads in dot computation, "
            << lLoadOps.size() << " local loads in dot computation";
    LDBG(message.str());
    return failure();
  }
  return success();
}

// Transform a loop into one Dot - Memory (ping - pong) clusters
// Each cluster, especially the Dot cluster is guarded with setprio(1->0) so
// each warp can complete the execution of the cluster without being
// interrupted. This is also supposed to be used with the numWarps=4 case where
// each SIMD runs two warps from different blocks and those two warps don't need
// to be synchronized together.
// Splitting loading A/B and interleave global/local load in order to prevent
// the stalls.
// sched.barriers with 0 mask were used to enforce the boundary of the
// high-level operations, inserting `setPrio` also has a same effect of
// instruction scheduling boundary, too.
void Pingponger::transformOnePPClusters(OpBuilder &builder, Location loc) {
  auto dotLoc = dotOps[0]->getPrevNode();
  // sched barrier to prevent memory ops from cross but leave other ops to be
  // scheduled across the barrier.
  auto preDotBar = builder.create<ROCDL::SchedBarrier>(loc, 1);
  updateOpInsertion(dotLoc);
  appendOp(preDotBar);

  // Memory cluster #0
  updateOpInsertion(lLoadOps[0]);
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, highPriority));
  moveOpAndPredecessorsUpSameBlock(gLoadOps[0]);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  moveOpAndPredecessorsUpSameBlock(lLoadOps[1]);
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, lowPriority));
  moveOpAndPredecessorsUpSameBlock(gLoadOps[1]);

  // Dot cluster #0
  updateOpInsertion(preDotBar);
  appendOpWithPrio(builder, dotOps[0], loc);
  // Add a remark for user feedback
  dotOps[0]->emitRemark() << "Performed one ping pong cluster transformation\n";
}

void Pingponger::genOffsetConstants(Location loc, OpBuilder &builder,
                                    unsigned numSlices, int64_t sliceWidth) {
  for (int i = 0; i < numSlices; i++) {
    int64_t offset = sliceWidth * i;
    constOffsets.push_back(
        builder.create<arith::ConstantIntOp>(loc, offset, 32));
  }
}

// Splits given local_loads for dot into multiple subviews and local_loads. This
// function tries to slice the local_load into the given number of the slices,
// generates ops when succeed, return fail() otherwise.
LogicalResult Pingponger::genLocalSlice(OpBuilder &builder, Value v,
                                        Attribute dotEncoding, unsigned opIdx,
                                        unsigned numSlices,
                                        int64_t sliceWidth) {
  // TODO: support transformed input to dot
  auto localLoad = v.getDefiningOp<ttg::LocalLoadOp>();
  if (!localLoad)
    return failure();
  auto memDesc = localLoad.getSrc();
  auto type = cast<ttg::MemDescType>(memDesc.getType());
  SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
  Type elementType = type.getElementType();
  int64_t kIdx = opIdx == 0 ? 1 : 0;
  shape[kIdx] = sliceWidth;
  // Each slice cannot be smaller than the smallest supported mfma width.
  if (sliceWidth < 16)
    return failure();
  auto dotOperandEnc = ttg::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding, kWidth);

  auto tensorType = RankedTensorType::get(shape, elementType, dotOperandEnc);

  return genLocalSliceHelper(builder, v, opIdx, numSlices, sliceWidth,
                             tensorType);
}

LogicalResult Pingponger::genLocalSliceScales(OpBuilder &builder, Value v,
                                              Attribute dotEncoding,
                                              unsigned opIdx,
                                              unsigned numSlices,
                                              int64_t sliceWidth) {
  auto localLoad = v.getDefiningOp<ttg::LocalLoadOp>();
  if (!localLoad)
    return failure();
  auto memDesc = localLoad.getSrc();
  auto type = cast<ttg::MemDescType>(memDesc.getType());
  SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
  Type elementType = type.getElementType();
  int64_t kIdx = opIdx == 0 ? 1 : 0;
  shape[kIdx] = sliceWidth;

  auto ll = mlir::triton::gpu::toLinearLayout(shape, dotEncoding);
  auto dotOperandEnc = ttg::LinearEncodingAttr::get(type.getContext(), ll);
  auto tensorType = RankedTensorType::get(shape, elementType, dotOperandEnc);

  return genLocalSliceHelper(builder, v, 0, numSlices, sliceWidth, tensorType);
}

LogicalResult Pingponger::genLocalSliceHelper(OpBuilder &builder, Value v,
                                              unsigned opIdx,
                                              unsigned numSlices,
                                              int64_t sliceWidth,
                                              RankedTensorType tensorType) {

  SmallVector<Operation *> slices;
  SmallVector<Operation *> subviews;

  auto localLoad = v.getDefiningOp<ttg::LocalLoadOp>();
  if (!localLoad)
    return failure();

  auto waitToken = localLoad.getToken();
  auto memDesc = localLoad.getSrc();
  auto type = cast<ttg::MemDescType>(memDesc.getType());
  SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
  Type elementType = type.getElementType();
  int64_t kIdx = opIdx == 0 ? 1 : 0;
  shape[kIdx] = sliceWidth;

  auto subviewDescType = ttg::MemDescType::get(
      shape, elementType, type.getEncoding(), type.getMemorySpace(),
      type.getMutableMemory(), type.getAllocShape());

  for (int i = 0; i < numSlices; i++) {
    SmallVector<Value> offsetsVal;
    SmallVector<int64_t> offsets = {0, 0};
    offsets[opIdx == 0 ? 1 : 0] = i;
    for (int64_t off : offsets) {
      offsetsVal.push_back(builder.create<arith::ConstantIntOp>(
          v.getLoc(), off * sliceWidth, 32));
    }
    Value newSmem = builder.create<ttg::MemDescSubviewOp>(
        v.getLoc(), subviewDescType, memDesc, offsetsVal);
    Value prefetchSlice = builder.create<ttg::LocalLoadOp>(
        v.getLoc(), tensorType, newSmem, waitToken);
    subviews.push_back(newSmem.getDefiningOp());
    slices.push_back(prefetchSlice.getDefiningOp());
  }
  subViewOps.push_back(subviews);
  loadSliceOps.push_back(slices);
  return success();
}

// Split dot into 'numSlices' pieces. This is required by pingpong scheduling
// when it needs to schedule multiple dot clusters. Calls genLocalSlice to
// create corresponding local_load slices.
LogicalResult Pingponger::sliceDot(OpBuilder &builder, Location loc,
                                   tt::DotOp op, unsigned numSlices) {
  builder.setInsertionPointToStart(forOp.getBody());
  auto typeB = op.getB().getType();
  auto shapeB = typeB.getShape();
  int64_t sliceWidth = shapeB[0] / numSlices;
  if (shapeB[0] % numSlices != 0)
    return failure();
  genOffsetConstants(loc, builder, numSlices, sliceWidth);
  builder.setInsertionPointAfter(useAsyncCopy ? asyncCopyOps[0] : gLoadOps[0]);
  auto dotEncoding = op.getType().getEncoding();
  if (genLocalSlice(builder, op.getA(), dotEncoding, 0, numSlices, sliceWidth)
          .failed() ||
      genLocalSlice(builder, op.getB(), dotEncoding, 1, numSlices, sliceWidth)
          .failed())
    return failure();

  // Clone dots to consume all the slices
  Operation *prevDot = op;
  for (int i = 0; i < numSlices; i++) {
    IRMapping mapping;
    mapping.map(op.getA(), loadSliceOps[0][i]->getResult(0));
    mapping.map(op.getB(), loadSliceOps[1][i]->getResult(0));
    if (i > 0)
      mapping.map(op.getC(), prevDot->getResult(0));
    auto newOp = builder.clone(*op, mapping);
    prevDot = newOp;
    dotSliceOps.push_back(newOp);
  }
  op->replaceAllUsesWith(prevDot);
  op->erase();
  for (auto loads : lLoadOps)
    loads->erase();
  return success();
}

LogicalResult Pingponger::sliceDotScaled(OpBuilder &builder, Location loc,
                                         tt::DotScaledOp op,
                                         unsigned numSlices) {
  builder.setInsertionPointToStart(forOp.getBody());
  auto typeB = op.getB().getType();
  auto typeScaleB = op.getBScale().getType();
  auto shapeB = typeB.getShape();
  auto shapeScaleB = typeScaleB.getShape();

  int64_t sliceWidth = shapeB[0] / numSlices;
  int64_t sliceScaleWidth = shapeScaleB[1] / numSlices;
  if (shapeB[1] % numSlices != 0)
    return failure();

  builder.setInsertionPointAfter(op);
  auto dotEncoding = op.getType().getEncoding();

  // Generate slices for operands A and B
  if (genLocalSlice(builder, op.getA(), dotEncoding, 0, numSlices, sliceWidth)
          .failed() ||
      genLocalSlice(builder, op.getB(), dotEncoding, 1, numSlices, sliceWidth)
          .failed())
    return failure();

  // Generate slices for scale tensors if they exist
  Value aScale = op.getAScale();
  Value bScale = op.getBScale();

  if (aScale) {
    if (genLocalSliceScales(builder, aScale,
                            op.getAScale().getType().getEncoding(), 0,
                            numSlices, sliceScaleWidth)
            .failed())
      return failure();
  }

  if (bScale) {
    if (genLocalSliceScales(builder, bScale,
                            op.getBScale().getType().getEncoding(), 0,
                            numSlices, sliceScaleWidth)
            .failed())
      return failure();
  }

  Operation *prevDot = op;
  for (int i = 0; i < numSlices; i++) {
    IRMapping mapping;
    mapping.map(op.getA(), loadSliceOps[0][i]->getResult(0));
    mapping.map(op.getB(), loadSliceOps[1][i]->getResult(0));

    // Map scale tensors if they exist
    if (aScale)
      mapping.map(op.getAScale(), loadSliceOps[2][i]->getResult(0));
    if (bScale)
      mapping.map(op.getBScale(), loadSliceOps[3][i]->getResult(0));

    if (i > 0)
      mapping.map(op.getC(), prevDot->getResult(0));

    auto newOp = builder.clone(*op, mapping);
    prevDot = newOp;
    dotSliceOps.push_back(newOp);
  }

  // Replace original op with the last slice and cleanup
  op->replaceAllUsesWith(prevDot);
  op->erase();
  for (auto loads : lLoadOps)
    loads->erase();
  return success();
}

// Transform a loop into four Dot - Memory (ping - pong) clusters
// This transform is useful when the original dot tile is too large that there's
// not enough registers to hold data for a Dot cluster. This path slices the dot
// into four pieces and pair with four clusters of reordered memory operations.
// There are multiple guards at the boundary of each cluster.
// (1) sched.barrier : with mask0 to prevent compiler backed from reordering
//  instructions across the boundary
// (2) gpu.barrier : ensures asymmetric synchronization at each point
// (3) setprio (1->0) : in order to avoid incoming warp overtaking resource
//  while the other warp is actively using it.
//
// Here's overview of the instruction clusters
// mem0: global load A, local load A(1/4), local load B(1/4)
// dot0: dot A(1/4) * B(1/4)
// mem1: global load B, local load A(2/4), local load B(2/4)
// dot1: dot A(2/4) * B(2/4)
// mem2: local load A(3/4, 4/4), local load B(3/4, 4/4)
// dot2: dot A(3/4) * B(3/4)
// mem3: local store A and B
// dot3: dot A(4/4) * B(4/4)

LogicalResult Pingponger::transformFourPPClusters(OpBuilder &builder,
                                                  Location loc) {
  // First, slice local_loads and dot into 4 parts
  if (sliceDot(builder, loc, dotOps[0], 4).failed())
    return failure();
  Operation *gLoadRhs = useAsyncCopy ? asyncCopyOps[1] : gLoadOps[1];
  builder.setInsertionPointAfter(gLoadRhs);
  // Reorder operations into four mem/dot clusters

  // mem0: global load A, local load A(1/4), local load B(1/4)
  // set insertion point at the last global_load where all the addresses are
  // ready to be used.
  updateOpInsertion(gLoadRhs);
  appendSlicedLoadAB(/*slice=*/0);
  appendClusterBarrier(builder, loc);

  // dot0 (1/4)
  appendOpWithPrio(builder, dotSliceOps[0], loc);
  appendClusterBarrier(builder, loc);

  // mem1: global load B, local load A(2/4), local load B(2/4)
  appendOp(gLoadRhs);
  if (useAsyncCopy) {
    appendOp(asyncCommitOps[1]);
  }
  appendSlicedLoadAB(/*slice=*/1);
  appendClusterBarrier(builder, loc);

  // dot1 (2/4)
  appendOpWithPrio(builder, dotSliceOps[1], loc);
  appendClusterBarrier(builder, loc);

  // mem2: local load A(3/4, 4/4), local load B(3/4, 4/4)
  appendSlicedLoadAB(/*slice=*/2);
  appendSlicedLoadAB(/*slice=*/3);
  appendClusterBarrier(builder, loc);

  // dot2 (3/4)
  appendOpWithPrio(builder, dotSliceOps[2], loc);
  appendClusterBarrier(builder, loc);

  // mem3: local store A and B
  // Matmul kernels may use the output of the dot product in another operation
  // before the local store (e.g. persistent matmul epilogue). To accommodate
  // such cases, we need to move the local store up in the loop.
  if (!useAsyncCopy) {
    moveOpAndPredecessorsUpSameBlock(lStoreOps[0]);
    moveOpAndPredecessorsUpSameBlock(lStoreOps[1]);
    appendClusterBarrier(builder, loc);
  } else {
    appendOp(asyncWaitOps[0]);
    appendOp(asyncWaitOps[1]);
  }
  // dot3 (4/4)
  appendOpWithPrio(builder, dotSliceOps[3], loc);

  // Move the cluster barrier to the end of the main loop.
  // This helps ensure that with persistent GEMMs the epilogue
  // and prologue aren't grouped into the same long cluster.
  updateOpInsertion(lastInsertedOp->getBlock()->getTerminator());
  prependClusterBarrier(builder, loc);

  // Add a remark for user feedback
  dotSliceOps[0]->emitRemark()
      << "Performed four ping pong cluster transformation\n";
  return success();
}

// Transform a loop into two Dot - Memory (ping - pong) clusters
// This is useful for the medium sized tile which doesn't fit to either one/four
// cluster scheduling.
LogicalResult Pingponger::transformTwoPPClusters(OpBuilder &builder,
                                                 Location loc) {
  // First, slice local_loads and dot into 2 parts
  if (sliceDot(builder, loc, dotOps[0], 2).failed())
    return failure();
  builder.setInsertionPointAfter(gLoadOps[1]);
  // Reorder operations into two mem/dot clusters

  // Memory cluster #0
  // interleave local_loads and global_loads to minimize the stalling
  // cycles, sched.barrier prevents backend from canceling the interleaved order
  updateOpInsertion(gLoadOps[1]);
  appendSlicedLoadAB(/*slice=*/0);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(gLoadOps[0]);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendSlicedLoadAB(/*slice=*/1);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(gLoadOps[1]);
  // The first cluster just fits into the two cluster pingpong and cannot
  // include wait of the local_load inserted by the gpu.barrier, using s.barrier
  // instead. backend will schedule the local memory fences later in the dot0
  // cluster.
  appendOp(builder.create<ROCDL::SBarrierOp>(loc));
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));

  // dot0 (1/2)
  appendOpWithPrio(builder, dotSliceOps[0], loc);
  appendClusterBarrier(builder, loc);

  // mem1: local store A and B
  // Matmul kernels may use the output of the dot product in another operation
  // before the local store (e.g. persistent matmul epilogue). To accommodate
  // such cases, we need to move the local store up in the loop.
  moveOpAndPredecessorsUpSameBlock(lStoreOps[0]);
  moveOpAndPredecessorsUpSameBlock(lStoreOps[1]);
  appendClusterBarrier(builder, loc);

  // dot1 (2/2)
  appendOpWithPrio(builder, dotSliceOps[1], loc);

  // Move the cluster barrier to the end of the main loop.
  // This helps ensure that with persistent GEMMs the epilogue
  // and prologue aren't grouped into the same long cluster.
  updateOpInsertion(lastInsertedOp->getBlock()->getTerminator());
  prependClusterBarrier(builder, loc);

  // Add a remark for user feedback
  dotSliceOps[0]->emitRemark()
      << "Performed two ping pong cluster transformation\n";
  return success();
}

// Fixme : document the scheduling.
// Assuming pipeliner already ordered the ops.
LogicalResult Pingponger::transformFAv3(OpBuilder &builder, Location loc) {
  if (asyncWaitOps.size() != 2) {
    return llvm::failure();
  }

  builder.setInsertionPointToStart(forOp.getBody());
  updateOpInsertion(dotOps[0]);
  prependOp(builder.create<ROCDL::SetPrioOp>(loc, lowPriority), false);

  // dot cluster 0 operations here.

  updateOpInsertion(asyncWaitOps[0]);
  prependOp(builder.create<ROCDL::SetPrioOp>(loc, highPriority), false);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));

  // mem cluster 0 operations here.

  updateOpInsertion(dotOps[1]);
  // below ops are inserted backward
  prependOp(builder.create<ROCDL::SetPrioOp>(loc, lowPriority), true);
  prependOp(builder.create<ROCDL::SBarrierOp>(loc), true);
  prependOp(builder.create<ROCDL::SchedBarrier>(loc, 0), true);

  // dot cluster 1 operations here.

  updateOpInsertion(asyncWaitOps[1]);
  prependOp(builder.create<ROCDL::SetPrioOp>(loc, highPriority), false);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));

  // mem cluster 1 operations here.

  updateOpInsertion(lastInsertedOp->getBlock()->getTerminator());
  prependOp(builder.create<ROCDL::SBarrierOp>(loc), true);
  prependOp(builder.create<ROCDL::SchedBarrier>(loc, 0), true);

  // Fixme: validate the case here?
  return success();
}

LogicalResult Pingponger::transformFP4s(OpBuilder &builder, Location loc) {
  // FIXME: support nonscale.
  if (lLoadOps.size() != 4)
    return failure();

  auto tokens = asyncWaitOps[0].getAsyncToken();
  Operation *aWait = asyncWaitOps[0];
  builder.setInsertionPointToStart(forOp.getBody());
  asyncWaitOps.clear();
  for (int i = 0; i < 2; i++) {
    auto newOp = builder.clone(*aWait);
    newOp->eraseOperand(3 - i);
    newOp->eraseOperand(1 - i);
    asyncWaitOps.push_back(cast<ttg::AsyncWaitOp>(newOp));
  }
  lLoadOps[0]->replaceUsesOfWith(aWait->getResult(0), asyncWaitOps[0]);
  lLoadOps[2]->replaceUsesOfWith(aWait->getResult(0), asyncWaitOps[0]);
  lLoadOps[1]->replaceUsesOfWith(aWait->getResult(0), asyncWaitOps[1]);
  lLoadOps[3]->replaceUsesOfWith(aWait->getResult(0), asyncWaitOps[1]);
  aWait->erase();

  builder.setInsertionPointAfter(dotSOps[0]);
  updateOpInsertion(dotSOps[0]);

  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(builder.create<ROCDL::SBarrierOp>(loc));
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(lLoadOps[0]);
  appendOp(lLoadOps[2]);

  appendOp(asyncWaitOps[1]);

  appendOp(asyncCopyOps[1]);
  appendOp(asyncCopyOps[3]);
  appendOp(asyncCommitOps[1]);
  appendOp(asyncCommitOps[3]);

  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(builder.create<ROCDL::SBarrierOp>(loc));
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));

  appendOp(lLoadOps[1]);
  appendOp(lLoadOps[3]);
  appendOp(dotSOps[0]);

  return success();
}

LogicalResult Pingponger::transformFP4(OpBuilder &builder, Location loc) {

  builder.setInsertionPointAfter(forOp);

  // FIXME: This is duplicated code, need to refactorize.
  auto i32ty = builder.getIntegerType(32);
  auto workIDX = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
  workIDX->moveBefore(forOp);
  builder.setInsertionPointAfter(workIDX);
  auto constZero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  auto constWarpSize = builder.create<arith::ConstantIntOp>(loc, 256, 32);
  auto warpIDX = builder.create<arith::DivSIOp>(loc, workIDX, constWarpSize);
  auto warpLow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               warpIDX, constZero);
  auto warpHigh = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                warpIDX, constZero);

  builder.setInsertionPointAfter(dotSOps[0]);

  if (sliceDotScaled(builder, loc, dotSOps[0], 4).failed())
    return failure();
  updateOpInsertion(dotSliceOps[0]);

  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(builder.create<tt::amdgpu::CondBarrierOp>(loc, warpLow));
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++)
      appendOp(subViewOps[i][j]);
    for (int i = 0; i < 4; i++)
      appendOp(loadSliceOps[i][j]);
    appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
    appendOp(dotSliceOps[j]);
  }

  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(builder.create<tt::amdgpu::CondBarrierOp>(loc, warpHigh));

  return success();
}

LogicalResult Pingponger::genScaleSlice(OpBuilder &builder, Value v,
                                        unsigned numSlices, int sliceDim) {
  // TODO: support transformed input to dot
  auto localLoad = v.getDefiningOp<ttg::LocalLoadOp>();
  if (!localLoad)
    return failure();
  auto memDesc = localLoad.getSrc();
  auto type = cast<ttg::MemDescType>(memDesc.getType());
  auto srcTy = cast<RankedTensorType>(v.getType());
  SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
  Type elementType = type.getElementType();
  int64_t sliceDimSize = shape[sliceDim];
  int64_t sliceWidth = sliceDimSize / numSlices;
  shape[sliceDim] = sliceWidth;

  // auto llEnc = cast<ttg::LinearEncodingAttr>(srcTy.getEncoding());
  // auto newLLEnc = triton::gpu::LinearEncodingAttr::get(builder.getContext(),
  //                                                      llEnc.getLinearLayout());
  auto sliceType =
      RankedTensorType::get(shape, elementType, srcTy.getEncoding());

  auto subviewDescType = ttg::MemDescType::get(
      shape, elementType, type.getEncoding(), type.getMemorySpace(),
      type.getMutableMemory(), type.getAllocShape());

  auto &slicedSubviews = subViewOps.emplace_back();
  auto &slicedLoads = loadSliceOps.emplace_back();
  for (int i = 0; i < numSlices; i++) {
    SmallVector<Value> offsetsVal;
    SmallVector<int64_t> offsets = {0, 0};
    offsets[sliceDim] = i;
    for (int64_t off : offsets) {
      offsetsVal.push_back(builder.create<arith::ConstantIntOp>(
          v.getLoc(), off * sliceWidth, 32));
    }
    auto newSmem = builder.create<ttg::MemDescSubviewOp>(
        v.getLoc(), subviewDescType, memDesc, offsetsVal);
    auto prefetchSlice = builder.create<ttg::LocalLoadOp>(
        v.getLoc(), sliceType, newSmem, localLoad.getToken());
    slicedSubviews.push_back(newSmem);
    slicedLoads.push_back(prefetchSlice);
  }
  return success();
}

SmallVector<Value> Pingponger::genSplitAsyncCopy(OpBuilder &builder, Value v,
                                                 unsigned numSlices,
                                                 unsigned sliceDim) {
  auto loc = v.getLoc();
  auto copyOp = cast<ttg::AsyncCopyGlobalToLocalOp>(v.getDefiningOp());
  auto srcTy = cast<RankedTensorType>(copyOp.getSrc().getType());

  SmallVector<int64_t> shape = to_vector(srcTy.getShape());
  int64_t sliceDimSize = shape[sliceDim];
  int64_t sliceWidth = sliceDimSize / numSlices;
  shape[sliceDim] = sliceWidth;

  auto splitTy =
      RankedTensorType::get(shape, srcTy.getElementType(), srcTy.getEncoding());
  auto memDesc = cast<ttg::MemDescType>(copyOp.getResult().getType());
  auto subviewDescType = ttg::MemDescType::get(
      {128, 128}, memDesc.getElementType(), memDesc.getEncoding(),
      memDesc.getMemorySpace(), memDesc.getMutableMemory(),
      memDesc.getAllocShape());

  SmallVector<Value> tokens;
  for (int i = 0; i < numSlices; i++) {
    SmallVector<int64_t> offsets(shape.size(), 0);
    offsets[sliceDim] = sliceWidth * i;
    Value slicedSrc = builder.create<triton::amdgpu::ExtractSliceOp>(
        loc, splitTy, copyOp.getSrc(), builder.getDenseI64ArrayAttr(offsets));

    SmallVector<Value> offsetVals;
    for (auto &o : offsets)
      offsetVals.push_back(builder.create<arith::ConstantIntOp>(loc, o, 32));

    Value subview = builder.create<ttg::MemDescSubviewOp>(
        copyOp.getLoc(), subviewDescType, copyOp.getResult(), offsetVals);

    Value slicedCopy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
        loc, slicedSrc, subview, copyOp.getMask(), copyOp.getOther(),
        copyOp.getCache(), copyOp.getEvict(), copyOp.getIsVolatile());
    Value commit = builder.create<ttg::AsyncCommitGroupOp>(
        loc, slicedCopy.getType(), ValueRange(slicedCopy));
    tokens.push_back(commit);
  }

  return tokens;
}

LogicalResult Pingponger::transformFP4mn(OpBuilder &builder, Location loc) {

  builder.setInsertionPoint(dotSOps[0]);

  // ------- Split AsyncCopy
  auto slicedATokens = genSplitAsyncCopy(builder, Value(asyncCopyOps[2]), 2, 0);
  auto slicedBTokens = genSplitAsyncCopy(builder, Value(asyncCopyOps[3]), 2, 1);
  if (slicedATokens.empty() || slicedBTokens.empty())
    return failure();

  // ------ Remove the old AsyncWait
  auto oldAsyncWait = asyncWaitOps[0];
  SmallVector<int64_t> tokenArgIds;
  SmallVector<Value> initialTokens;
  for (auto inputToken : oldAsyncWait->getOperands()) {
    auto blockArg = dyn_cast<BlockArgument>(inputToken);
    if (!blockArg) {
      return failure();
    }
    tokenArgIds.push_back(blockArg.getArgNumber());
    initialTokens.push_back(forOp.getInitArgs()[blockArg.getArgNumber() - 1]);
  }

  OpBuilder beforeFor(forOp);
  auto waitBeforeLoop =
      beforeFor.create<ttg::AsyncWaitOp>(loc, initialTokens, 0);

  // --------
  // Overall we add 7 new tokens to the loop. 2x OperandA, 2x Operand B, 1x
  // ScaleA, 1x ScaleB, 1x for last AsyncWait in loop

  // -------- Patch initial loop args with the token from the wait before the
  // loop
  Value foundToken;
  for (auto initialArg : forOp.getInitArgs()) {
    if (initialArg.getType() == slicedATokens[0].getType()) {
      foundToken = initialArg;
    }
  }

  SmallVector<Value> initArgs(forOp.getInitArgs());
  SmallVector<Value> newArgs(7, waitBeforeLoop);
  // forOp.getInitArgsMutable().append(newArgs);
  unsigned startIndex = forOp.getBody()->getNumArguments();
  auto newBlockArgs = addIterArgsToLoop(builder, forOp, newArgs);

  SmallVector<Value> loopCarriedTokensSliceA = {
      forOp.getBody()->getArgument(startIndex),
      forOp.getBody()->getArgument(startIndex + 1),
  };
  SmallVector<Value> loopCarriedTokensSliceB = {
      forOp.getBody()->getArgument(startIndex + 2),
      forOp.getBody()->getArgument(startIndex + 3),
  };
  Value loopCarriedTokensScaleA = forOp.getBody()->getArgument(startIndex + 4);
  Value loopCarriedTokensScaleB = forOp.getBody()->getArgument(startIndex + 5);
  Value loopCarriedAsyncWaitToken =
      forOp.getBody()->getArgument(startIndex + 6);

  // ------ Cleanup old AsyncWait, we just replace it with one of its operands
  // and will update each LocalLoad later
  asyncWaitOps[0].getRetToken().replaceAllUsesWith(
      Value(forOp.getBody()->getArgument(tokenArgIds.front())));
  asyncWaitOps[0]->erase();

  // ------- Slice scaleA
  auto aScale = dotSOps[0].getAScale();
  if (failed(genScaleSlice(builder, aScale, 2, 0))) {
    return failure();
  }
  Value sliceScaleA1 = loadSliceOps[0][0]->getResult(0);
  Value sliceScaleA2 = loadSliceOps[0][1]->getResult(0);

  // ------- Slice scaleB
  auto bScale = dotSOps[0].getBScale();
  if (failed(genScaleSlice(builder, bScale, 2, 0))) {
    return failure();
  }
  Value sliceScaleB1 = loadSliceOps[1][0]->getResult(0);
  Value sliceScaleB2 = loadSliceOps[1][1]->getResult(0);

  // ------- Slice A
  auto aOperand = dotSOps[0].getA();
  if (failed(genScaleSlice(builder, aOperand, 2, 0))) {
    return failure();
  }
  Value sliceOperandA1 = loadSliceOps[2][0]->getResult(0);
  Value sliceOperandA2 = loadSliceOps[2][1]->getResult(0);

  // ------- Slice B
  auto bOperand = dotSOps[0].getB();
  if (failed(genScaleSlice(builder, bOperand, 2, 1))) {
    return failure();
  }
  Value sliceOperandB1 = loadSliceOps[3][0]->getResult(0);
  Value sliceOperandB2 = loadSliceOps[3][1]->getResult(0);

  // We have to adjust the tokens since we now have 2 separate tokens

  // -------- Slice Accumualator into 4 pieces
  Value acc = dotSOps[0].getC();
  auto accTy = cast<RankedTensorType>(acc.getType());
  auto sliceAccTy = RankedTensorType::get({128, 128}, accTy.getElementType(),
                                          accTy.getEncoding());
  auto slicedAcc0 = builder.create<triton::amdgpu::ExtractSliceOp>(
      loc, sliceAccTy, acc, builder.getDenseI64ArrayAttr({0, 0}));
  auto slicedAcc1 = builder.create<triton::amdgpu::ExtractSliceOp>(
      loc, sliceAccTy, acc, builder.getDenseI64ArrayAttr({0, 128}));
  auto slicedAcc2 = builder.create<triton::amdgpu::ExtractSliceOp>(
      loc, sliceAccTy, acc, builder.getDenseI64ArrayAttr({128, 0}));
  auto slicedAcc3 = builder.create<triton::amdgpu::ExtractSliceOp>(
      loc, sliceAccTy, acc, builder.getDenseI64ArrayAttr({128, 128}));

  // --------- Slice dot

  auto scliedResTy = sliceAccTy;
  Value dot1 = builder.create<triton::DotScaledOp>(
      loc, sliceAccTy, sliceOperandA1, sliceOperandB1, slicedAcc0, sliceScaleA1,
      sliceScaleB1, dotSOps[0].getAElemTypeAttr(),
      dotSOps[0].getBElemTypeAttr(), builder.getBoolAttr(false));
  Value dot2 = builder.create<triton::DotScaledOp>(
      loc, sliceAccTy, sliceOperandA1, sliceOperandB2, slicedAcc1, sliceScaleA1,
      sliceScaleB2, dotSOps[0].getAElemTypeAttr(),
      dotSOps[0].getBElemTypeAttr(), builder.getBoolAttr(false));
  Value dot3 = builder.create<triton::DotScaledOp>(
      loc, sliceAccTy, sliceOperandA2, sliceOperandB1, slicedAcc2, sliceScaleA2,
      sliceScaleB1, dotSOps[0].getAElemTypeAttr(),
      dotSOps[0].getBElemTypeAttr(), builder.getBoolAttr(false));
  Value dot4 = builder.create<triton::DotScaledOp>(
      loc, sliceAccTy, sliceOperandA2, sliceOperandB2, slicedAcc3, sliceScaleA2,
      sliceScaleB2, dotSOps[0].getAElemTypeAttr(),
      dotSOps[0].getBElemTypeAttr(), builder.getBoolAttr(false));

  // -----------------------------------------------------------
  // Schedule values
  // -----------------------------------------------------------
  builder.setInsertionPointAfter(dotSOps[0]);
  updateOpInsertion(dotSOps[0]);

  Value scaleACommit = asyncCommitOps[0];
  Value scaleBCommit = asyncCommitOps[1];
  Value lastAsyncWaitToken = loopCarriedAsyncWaitToken;

  auto appendLocalLoadOp = [&](Operation *localLoad) {
    appendOp(localLoad);
    cast<ttg::LocalLoadOp>(localLoad).getTokenMutable().assign(
        lastAsyncWaitToken);
  };

  auto appendAsyncWaitOp = [&](ttg::AsyncWaitOp asyncWait) {
    appendOp(asyncWait);
    lastAsyncWaitToken = asyncWait;
  };

  // For AsyncCopies we always need to move the AsyncCopy and the CommitGroup
  // A0
  appendOp(slicedATokens[0].getDefiningOp()->getOperand(0).getDefiningOp());
  appendOp(slicedATokens[0].getDefiningOp());
  // B0
  appendOp(slicedBTokens[0].getDefiningOp()->getOperand(0).getDefiningOp());
  appendOp(slicedBTokens[0].getDefiningOp());
  // SA
  appendOp(asyncCopyOps[0]);
  for (auto user : asyncCopyOps[0]->getUsers())
    appendOp(user);
  // SB
  appendOp(asyncCopyOps[1]);
  for (auto user : asyncCopyOps[1]->getUsers())
    appendOp(user);

  // Local Loads A0, B0, SA0, SB0
  appendLocalLoadOp(sliceOperandA1.getDefiningOp());
  appendLocalLoadOp(sliceOperandB1.getDefiningOp());
  appendLocalLoadOp(sliceScaleA1.getDefiningOp());
  appendLocalLoadOp(sliceScaleB1.getDefiningOp());
  // TODO add async wait B1
  appendAsyncWaitOp(
      builder.create<ttg::AsyncWaitOp>(loc, loopCarriedTokensSliceB[1], 0));

  appendOp(dot1.getDefiningOp());
  // TODO add barrier
  // B1
  appendOp(slicedBTokens[1].getDefiningOp()->getOperand(0).getDefiningOp());
  appendOp(slicedBTokens[1].getDefiningOp());

  // Local Loads B1, SB1
  appendLocalLoadOp(sliceOperandB2.getDefiningOp());
  appendLocalLoadOp(sliceScaleB2.getDefiningOp());

  // AsyncWait A1
  appendAsyncWaitOp(
      builder.create<ttg::AsyncWaitOp>(loc, loopCarriedTokensSliceA[1], 0));

  appendOp(dot2.getDefiningOp());

  // TODO barrier

  // A1
  appendOp(slicedATokens[1].getDefiningOp()->getOperand(0).getDefiningOp());
  appendOp(slicedATokens[1].getDefiningOp());

  // Local loads A1, SA1
  appendLocalLoadOp(sliceOperandA2.getDefiningOp());
  appendLocalLoadOp(sliceScaleA2.getDefiningOp());

  // AsyncWait A0, B0, SA, SB, note we do *not* use the loop carried tokens here
  appendAsyncWaitOp(builder.create<ttg::AsyncWaitOp>(
      loc,
      // ValueRange{loopCarriedTokensSliceA[0], loopCarriedTokensSliceB[0],
      //            loopCarriedTokensScaleA, loopCarriedTokensScaleB},
      ValueRange{slicedATokens[0], slicedBTokens[0], scaleACommit,
                 scaleBCommit},
      0));

  appendOp(dot3.getDefiningOp());
  appendOp(dot4.getDefiningOp());

  // Concat dot results and replace and erase old dot
  auto dotConcat = builder.create<triton::amdgpu::ConcatOp>(
      loc, accTy, ValueRange{dot1, dot2, dot3, dot4});
  dotSOps[0]->replaceAllUsesWith(dotConcat);
  dotSOps[0].erase();

  // ----------------- Patch the yield to pass the tokens of all sliced loads
  // and the last AsyncWait
  auto yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value> operands(yieldOp->getOperands());
  operands.append(slicedATokens.begin(), slicedATokens.end());
  operands.append(slicedBTokens.begin(), slicedBTokens.end());
  // ScaleA (commit group)
  operands.push_back(scaleACommit);
  // ScaleB (commit group)
  operands.push_back(scaleBCommit);
  operands.push_back(lastAsyncWaitToken);

  OpBuilder builder2(yieldOp);
  builder2.create<scf::YieldOp>(yieldOp->getLoc(), operands);
  yieldOp->erase();

  // Cleanup old AsyncCopies, we just rewrite the token to use the first slice
  auto commitA = *asyncCopyOps[2]->getUsers().begin();
  commitA->replaceAllUsesWith(slicedATokens[0].getDefiningOp());
  commitA->erase();
  asyncCopyOps[2]->erase();
  auto commitB = *asyncCopyOps[3]->getUsers().begin();
  commitB->replaceAllUsesWith(slicedBTokens[0].getDefiningOp());
  commitB->erase();
  asyncCopyOps[3]->erase();

  // asyncCopyOps[2]->erase();
  // asyncCopyOps[3]->erase();

  // oldCommit->erase();
  // asyncLoadA->erase();

  //  ::mlir::triton::ScaleDotElemTypeAttr a_elem_type,
  //  ::mlir::triton::ScaleDotElemTypeAttr b_elem_type, ::mlir::BoolAttr
  //  fastMath, ::mlir::BoolAttr lhs_k_pack = nullptr, ::mlir::BoolAttr
  //  rhs_k_pack = nullptr);

  // --------- Debug concat everything back together
  // auto concatScaleA = builder.create<triton::amdgpu::ConcatOp>(
  //     loc, aScale.getType(), ValueRange{sliceScaleA1, sliceScaleA2});
  // dotSOps[0].getAScaleMutable().assign(concatScaleA);
  // auto concatScaleB = builder.create<triton::amdgpu::ConcatOp>(
  //     loc, bScale.getType(), ValueRange{sliceScaleB1, sliceScaleB2});
  // dotSOps[0].getBScaleMutable().assign(concatScaleB);
  // auto concatOperandA = builder.create<triton::amdgpu::ConcatOp>(
  //     loc, aOperand.getType(), ValueRange{sliceOperandA1, sliceOperandA2});
  // dotSOps[0].getAMutable().assign(concatOperandA);
  // auto concatOperandB = builder.create<triton::amdgpu::ConcatOp>(
  //     loc, bOperand.getType(), ValueRange{sliceOperandB1, sliceOperandB2});
  // dotSOps[0].getBMutable().assign(concatOperandB);
  // auto accConcat = builder.create<triton::amdgpu::ConcatOp>(
  //     loc, accTy, ValueRange{slicedAcc0, slicedAcc1, slicedAcc2,
  //     slicedAcc3});
  // dotSOps[0].getCMutable().assign(accConcat);

  return success(0);

  builder.setInsertionPointAfter(forOp);

  // FIXME: This is duplicated code, need to refactorize.
  auto i32ty = builder.getIntegerType(32);
  auto workIDX = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
  workIDX->moveBefore(forOp);
  builder.setInsertionPointAfter(workIDX);
  auto constZero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  auto constWarpSize = builder.create<arith::ConstantIntOp>(loc, 256, 32);
  auto warpIDX = builder.create<arith::DivSIOp>(loc, workIDX, constWarpSize);
  auto warpLow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               warpIDX, constZero);
  auto warpHigh = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                warpIDX, constZero);

  builder.setInsertionPointAfter(dotSOps[0]);

  if (sliceDotScaled(builder, loc, dotSOps[0], 4).failed())
    return failure();
  updateOpInsertion(dotSliceOps[0]);

  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(builder.create<tt::amdgpu::CondBarrierOp>(loc, warpLow));
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++)
      appendOp(subViewOps[i][j]);
    for (int i = 0; i < 4; i++)
      appendOp(loadSliceOps[i][j]);
    appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
    appendOp(dotSliceOps[j]);
  }

  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(builder.create<tt::amdgpu::CondBarrierOp>(loc, warpHigh));

  return success();
}

// This function wraps forOp with cond_barrier. First, hold half of the warps
// (warpHigh) in a block before the loop so the barriers in the loop synchronize
// warps at the different point per the warp groups. After the loop, hold
// proceeding warps (warpLow) by calling cond_barrier on them.
void Pingponger::addAsymmetricSyncToLoop(OpBuilder &builder, Location loc) {
  builder.setInsertionPointAfter(forOp);
  // Set barrier before starting the loop. This resolves any remaining required
  // synchronization before beginning the specialized asymmetric
  // synchronization.
  auto preBarrier = builder.create<gpu::BarrierOp>(loc);
  preBarrier->moveBefore(forOp);
  builder.setInsertionPointAfter(preBarrier);

  // Insert condbarrier::second_half before starting the loop
  auto i32ty = builder.getIntegerType(32);
  auto workIDX = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
  auto constZero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  auto constWarpSize = builder.create<arith::ConstantIntOp>(loc, 256, 32);
  auto warpIDX = builder.create<arith::DivSIOp>(loc, workIDX, constWarpSize);
  auto warpLow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               warpIDX, constZero);
  auto warpHigh = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                warpIDX, constZero);
  auto condBarrierHigh =
      builder.create<tt::amdgpu::CondBarrierOp>(loc, warpHigh);

  // Insert condbarrier::first_half after the end of the loop
  builder.setInsertionPointAfter(forOp);
  auto condBarrierLow = builder.create<tt::amdgpu::CondBarrierOp>(loc, warpLow);
}

void Pingponger::getDotPingponged() {
  if (numStages != 2 && numStages != 4) {
    std::stringstream message;
    message << "All ping pong scheduling requires 2 or 4 stages. Found "
            << numStages << " stages";
    LDBG(message.str());
    return;
  }

  OpBuilder builder(forOp);
  MLIRContext *ctx = forOp.getContext();
  Location loc = forOp.getLoc();

  forOp->walk([&](Operation *op) {
    if (auto gLoad = dyn_cast<tt::LoadOp>(op))
      gLoadOps.push_back(gLoad);
    if (auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op))
      asyncCopyOps.push_back(asyncCopy);
    else if (auto lLoad = dyn_cast<ttg::LocalLoadOp>(op)) {
      // This scheduling doesn't help hiding intra-warp latency. So, we only
      // collect local_load ops that are software pipelined, which means their
      // source is from loop carried values
      auto src = lLoad.getSrc();
      if (auto arg = mlir::dyn_cast<BlockArgument>(src))
        if (auto tiedLoopInit = forOp.getTiedLoopInit(arg))
          if (tiedLoopInit->get())
            lLoadOps.push_back(lLoad);
    } else if (auto lStore = dyn_cast<ttg::LocalStoreOp>(op))
      lStoreOps.push_back(lStore);
    else if (auto pingpongDot = dyn_cast<tt::DotOp>(op)) {
      if (pingpongDot.getType().getRank() == 2)
        dotOps.push_back(pingpongDot);
    } else if (auto pingpongDot = dyn_cast<tt::DotScaledOp>(op)) {
      dotSOps.push_back(pingpongDot);
    } else if (auto asyncOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      asyncCopyOps.push_back(asyncOp);
    } else if (auto asyncCommitGroupOp =
                   dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
      asyncCommitOps.push_back(asyncCommitGroupOp);
    } else if (auto asyncOp = dyn_cast<ttg::AsyncWaitOp>(op))
      asyncWaitOps.push_back(asyncOp);
  });

  // Fixme : use proper condition to identify FAv3
  if (numStages == 4 && dotOps.size() == 2) {
    if (transformFAv3(builder, loc).failed()) {
      LDBG("Encountered failure when trying to execute the FAv3 ping pong "
           "cluster transformation");
      return;
    }
    addAsymmetricSyncToLoop(builder, loc);
    return;
  }

  // Currently, pingpong scheduling is known as helpful under limited condition.
  // Individual conditions are checked while collecting each operation such as
  // software pipelining and dot rank=2. Also only accept the for-loop with
  // supported combination of operations because this transformation is very
  // tightly scheduling the latencies.

  // FIXME: get better condition to enable pingpong either for dot or for
  // dot_scaled
  int64_t numOfDotLikeOps = dotSOps.size() + dotOps.size();
  if (numOfDotLikeOps != 1) {
    LDBG("Only handle a single of either dot or dot_scaled op");
    return;
  }
  int64_t gloadSize = useAsyncCopy ? asyncCopyOps.size() : gLoadOps.size();
  int64_t dotSize = dotSOps.size() > 0 ? dotSOps.size() : dotOps.size();
  if ((gloadSize < 2 || lLoadOps.size() < 2 || dotSize != 1)) {
    std::stringstream message;
    message << "Unable to match ping pong scheduling pattern. Details: "
            << gloadSize << " global loads, " << lLoadOps.size()
            << " local loads, " << dotSize << " dot products";
    LDBG(message.str());
    return;
  }

  // FIXME: place tile size restriction here and obtain kWidth
  if (dotSOps.size() == 1 && numWarps == 8 && numStages == 2 &&
      asyncCopyOps.size() > 0) {
    auto dotSType = dotSOps[0].getType();
    auto dotSShape = dotSType.getShape();
    auto aType = dotSOps[0].getA().getType();
    auto aShape = aType.getShape();
    auto elemWidth = aType.getElementTypeBitWidth();
    int64_t tileSize = dotSShape[0] * dotSShape[1] * aShape[1];

    // 256x256x256 (128xi8)
    if (tileSize == 8388608 && aShape[0] == 256 && aShape[1] == 128 &&
        elemWidth == 8) {
      kWidth = 16;
      if (transformFP4mn(builder, dotSOps[0]->getLoc()).failed()) {
        LDBG("Encountered failure when trying to execute the two ping pong "
             "cluster transformation");
        return;
      }
    }
    // 128x128x512 (256xi8)
    else if (tileSize == 4194304 && aShape[0] == 128 && aShape[1] == 256 &&
             elemWidth == 8) {
      if (transformFP4s(builder, dotSOps[0]->getLoc()).failed()) {
        LDBG("Encountered failure when trying to execute the two ping pong "
             "cluster transformation");
        return;
      }
    }

    addAsymmetricSyncToLoop(builder, loc);
    return;
  } else if (dotSOps.size() == 1)
    return;

  // Determine if we have a persistent GEMM. This will decide how we interpret
  // any memory operations that we find in conditionals.
  auto assumeNotTaken = isPersistentGemm(dotOps.size());

  // Compute tile size, kWidth, and mfma type.
  auto dotType = dotOps[0].getType();
  auto dotShape = dotType.getShape();
  auto aType = dotOps[0].getA().getType();
  auto aShape = aType.getShape();
  auto elemWidth = aType.getElementTypeBitWidth();
  int64_t tileSize = dotShape[0] * dotShape[1] * aShape[1] * elemWidth;

  const int64_t minTile = 262144;      // e.g. 32x128x64x16bit
  const int64_t smallTile = 16777216;  // e.g. 128x128x64x16bit
  const int64_t mediumTile = 33554432; // smallTile x 2
  const int64_t largeTile = 67108864;  // e.g. 256x256x64x16bit

  auto encoding = cast<RankedTensorType>(aType).getEncoding();
  auto srcEncoding = cast<ttg::DotOperandEncodingAttr>(encoding);
  kWidth = srcEncoding.getKWidth();
  auto mfmaEncoding = cast<ttg::AMDMfmaEncodingAttr>(srcEncoding.getParent());
  SmallVector<int64_t> intShape;
  intShape.push_back(mfmaEncoding.getMDim());
  intShape.push_back(mfmaEncoding.getNDim());

  if (dotOps.size() == 1 && useAsyncCopy) {
    if (numWarps != 8) {
      LDBG("Currently only support num_warp=8 for async PP");
      return;
    }
    if (tileSize != largeTile || aShape[1] != 64 || elemWidth != 16) {
      LDBG("Only support tile size of 256x256x64 tile size for async PP");
      return;
    }

    auto encoding = cast<RankedTensorType>(aType).getEncoding();
    auto srcEncoding = cast<ttg::DotOperandEncodingAttr>(encoding);
    kWidth = srcEncoding.getKWidth();
    auto mfmaEncoding = cast<ttg::AMDMfmaEncodingAttr>(srcEncoding.getParent());
    if (mfmaEncoding.getMDim() != 16 && mfmaEncoding.getNDim() != 16 &&
        kWidth != 8) {
      LDBG("Only support 16x16 intrinsic and kWidth=8 for async PP");
    }

    DenseSet<ttg::AsyncCopyGlobalToLocalOp> dotGlobalLoads;
    DenseSet<ttg::LocalLoadOp> dotLocalLoads;
    DenseSet<ttg::AsyncCommitGroupOp> dotAsyncCommitGroups;
    DenseSet<ttg::AsyncWaitOp> dotAsyncWaits;
    determineDotAsyncMemoryOps(dotOps[0], dotGlobalLoads, dotLocalLoads,
                               dotAsyncCommitGroups, dotAsyncWaits);
    if (failed(pruneDotAsyncMemoryOps(dotGlobalLoads, dotLocalLoads,
                                      dotAsyncCommitGroups, dotAsyncWaits,
                                      assumeNotTaken))) {
      std::stringstream message;
      message << "Failed to match ping pong scheduling pattern and prune async "
                 "memory ops.";
      LDBG(message.str());
      return;
    }
    if (transformFourPPClusters(builder, dotOps[0]->getLoc()).failed()) {
      LDBG("Encountered failure when trying to execute the four ping pong "
           "cluster transformation");
      return;
    }
    addAsymmetricSyncToLoop(builder, loc);
    return;
  }
  // The existing code depends on the loads being targeted being safe to move,
  // which will not hold if we do not properly have a GEMM. As a result, we
  // filter the associated load operations to only those that are associated
  // // with the GEMM.
  DenseSet<tt::LoadOp> dotGlobalLoads;
  DenseSet<ttg::LocalLoadOp> dotLocalLoads;
  DenseSet<ttg::LocalStoreOp> dotLocalStores;
  determineDotMemoryOps(dotOps[0], dotGlobalLoads, dotLocalLoads,
                        dotLocalStores);
  if (failed(pruneDotMemoryOps(dotGlobalLoads, dotLocalLoads, dotLocalStores,
                               assumeNotTaken))) {
    std::stringstream message;
    message << "Failed to match ping pong scheduling pattern and prune "
               "memory ops.";
    LDBG(message.str());
    return;
  }

  // Pingpong scheduling tries to form two different types of the instruction
  // clusters, i.e., Dot clusters and Memory clusters. While each SIMD has
  // two concurrent warps, both warps can execute a different type of
  // instruction cluster in parallel. Here are currently available patterns,
  // more patterns could be added later.
  //
  // (1) One Dot-Memory (ping-pong) cluster
  //  :Ideal to support small tile size e.g., 128x128x64_FP16. Where amount
  //   of the data used per each iteration is small enough and not causing
  //   local_load waiting or register spilling. Currently used for numWarps=4
  //   case where SIMD can hold two warps from different blocks.
  //
  // (2) Four Dot-Memory (ping-pongx4) clusters
  //  :Useful for the larger tile size e.g., 256x256x64_FP16. Clustering
  //   the Dot instruction (mfma) all together without fetching data requires
  //   GPU to hold all the data for the calculation. Such large tile size
  //   exceeds the amount of register GPU has so, we need to split the dot
  //   into several pieces.
  //
  // (3) Two Dot-Memory (ping-pongx2) clusters
  //  :Covers medium sized tile e.g., 256x128x64_FP16. Different tile size may
  //  require different scheduling pattern because the loop consists of
  //  different amount of memory transfer and dot operation. This scheduling
  //  support the tile sizes not supported by above two methods.
  //
  // N.B., Tile size smaller than 128x128x64_FP16 is likely not compute-bound
  // that pingpong scheduling doesn't help much.

  if (numWarps == 4) { // Pingpong between warps from different blocks
    // Transform a loop with small tile size.
    // We've observed that this small tile size spent almost equivalent cycle
    // times for issuing the memory operations and issuing dot operations,
    // smaller tile sizes are not likely to get any advantage from current dot
    // centric pingpong scheduling.
    if (tileSize <= smallTile && tileSize >= minTile)
      transformOnePPClusters(builder, loc);
    // numWarps=4 doesn't need asymmetric sync, return.
    return;
  } else if (numWarps == 8) { // Pingpong between warps from the same block
    if (lStoreOps.size() != 2) {
      std::stringstream message;
      message << "Unable to match ping pong slicing pattern. Details: "
              << lStoreOps.size() << " local stores in dot computation ";
      LDBG(message.str());
      return;
    }
    // Transform a loop where the tile size requires dots to be sliced
    if (tileSize == mediumTile) {
      if (transformTwoPPClusters(builder, dotOps[0]->getLoc()).failed()) {
        LDBG("Encountered failure when trying to execute the two ping pong "
             "cluster transformation");
        return;
      }
    } else if (tileSize >= largeTile) {
      // Avoid known register spilling. i.e., mfma16x16x16 & largetile & kpack>1
      if (intShape[0] == 16 && intShape[1] == 16 && kWidth == 8) {
        LDBG("Reached known register spilling case, skip pingpong scheduling");
        return;
      }
      if (transformFourPPClusters(builder, dotOps[0]->getLoc()).failed()) {
        LDBG("Encountered failure when trying to execute the four ping pong "
             "cluster transformation");
        return;
      }
    } else
      return;

    // Let half of the warps start the loop first and the others follow later
    // but in the synchronized way. This can be accomplished by calling
    // cond_barrier for the second half before the beginning of the loop so they
    // can wait until the first half hit the first barrier in the loop. Also
    // need to call cond_barrier for the first_half after exiting the loop, so
    // all warps can converge again.
    addAsymmetricSyncToLoop(builder, loc);
  }
}

class TritonAMDGPUBlockPingpongPass
    : public TritonAMDGPUBlockPingpongBase<TritonAMDGPUBlockPingpongPass> {
public:
  TritonAMDGPUBlockPingpongPass() = default;
  TritonAMDGPUBlockPingpongPass(int32_t numStages, bool useAsyncCopy) {
    this->numStages = numStages;
    this->useAsyncCopy = useAsyncCopy;
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    for (auto funcOp : m.getOps<tt::FuncOp>()) {
      funcOp.walk([&](scf::ForOp forOp) {
        Pingponger pingponger(forOp, ttg::lookupNumWarps(forOp), numStages,
                              useAsyncCopy);
        pingponger.getDotPingponged();
      });
    }
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createTritonAMDGPUBlockPingpongPass(int32_t numStages,
                                          bool useAsyncCopy) {
  return std::make_unique<TritonAMDGPUBlockPingpongPass>(numStages,
                                                         useAsyncCopy);
}
