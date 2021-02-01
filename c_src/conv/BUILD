load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_cuda_or_rocm",)

package(default_visibility=["//visibility:private"])

cc_binary(
  name = "libconv.so",
  srcs = ["conv.cc"] + glob(["erts/**/*.h"]),
  deps = if_cuda_or_rocm([
    "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
    ]) + [
    "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
    "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
    "@org_tensorflow//tensorflow/core/framework:allocator",
    "@org_tensorflow//tensorflow/compiler/xla:cpu_function_runtime",
    "@org_tensorflow//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
    "@org_tensorflow//tensorflow/core:lib",
    "@org_tensorflow//tensorflow/core/common_runtime:bfc_allocator",
    "@org_tensorflow//tensorflow/core/common_runtime/gpu:gpu_bfc_allocator",
    "@org_tensorflow//tensorflow/core/common_runtime/device:device_mem_allocator",
    "@org_tensorflow//tensorflow/stream_executor:tf_allocator_adapter",
  ],
  linkopts = ["-shared"],
  linkshared = 1,
)