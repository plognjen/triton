// RUN: triton-opt %s -split-input-file --tritonamdgpu-enable-moe-lds-bypass --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx940 matrix-instruction-size=0' | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonamdgpu-enable-moe-lds-bypass --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx940 matrix-instruction-size=16' | FileCheck %s
// RUN: (triton-opt %s -split-input-file --mlir-pass-pipeline-crash-reproducer=/dev/null --tritonamdgpu-enable-moe-lds-bypass --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx940 matrix-instruction-size=32' 2>&1 || true) | FileCheck %s --check-prefix=FORCE32

#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK: #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 16]
// CHECK-LABEL: kernel
// FORCE32-NOT: kernel
// FORCE32: moe lds bypass optimization supports only mfma 16x16
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @kernel(
      %arg0: tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<32x128x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x128xf32, #blocked>
    tt.store %arg2, %1 : tensor<32x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
