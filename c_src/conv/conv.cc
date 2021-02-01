#include "tensorflow/compiler/xla/conv/erts/erl_nif.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device/device_mem_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/mem.h"

static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM load_info) {
  return 0;
}

ERL_NIF_TERM conv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  stream_executor::Platform* platform =
    xla::PlatformUtil::GetPlatform(std::string("CUDA")).ConsumeValueOrDie();

  xla::LocalClientOptions options;
  options.set_platform(platform);
  options.set_number_of_replicas(1);
  options.set_intra_op_parallelism_threads(-1);

  xla::LocalClient* client =
    xla::ClientLibrary::GetOrCreateLocalClient(options).ConsumeValueOrDie();

  stream_executor::StreamExecutor* executor =
      client->backend().stream_executor(1).ConsumeValueOrDie();

  auto compute_stream = std::make_unique<stream_executor::Stream>(executor);
  compute_stream->Init();

  tensorflow::int64 free_memory;
  tensorflow::int64 total_memory;
  if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
    return enif_make_atom(env, "error");
  }

  size_t allocator_memory = free_memory * 0.9;

  auto sub_allocator = std::make_unique<tensorflow::DeviceMemAllocator>(
    /*stream_executor=*/executor,
    /*platform_id=*/tensorflow::PlatformDeviceId(1),
    /*use_unified_memory=*/true,
    /*alloc_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>(),
    /*free_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>());

  auto gpu_bfc_allocator = std::make_unique<tensorflow::BFCAllocator>(
    /*sub_allocator=*/sub_allocator.release(),
    /*total_memory=*/allocator_memory,
    /*allow_growth=*/false,
    /*name=*/absl::StrCat("GPU_", 1, "_bfc"),
    /*garbage_collection=*/false);

  auto allocator = std::make_unique<stream_executor::TfAllocatorAdapter>(gpu_bfc_allocator.release(), compute_stream.get());

  xla::XlaBuilder* builder = new xla::XlaBuilder("conv");
  xla::Shape input_shape = xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {32, 1, 9, 9});
  xla::Shape kernel_shape = xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {32, 1, 7, 7});

  float low = 0.0;
  float high = 1.0;

  xla::XlaOp zero = xla::ConstantR0(builder, low);
  xla::XlaOp one = xla::ConstantR0(builder, high);

  xla::XlaOp inp = xla::RngUniform(zero, one, input_shape);
  xla::XlaOp kernel = xla::RngUniform(zero, one, kernel_shape);

  xla::ConvolutionDimensionNumbers dimension_numbers;
  dimension_numbers.set_input_batch_dimension(0);
  dimension_numbers.set_input_feature_dimension(1);
  dimension_numbers.add_input_spatial_dimensions(2);
  dimension_numbers.add_input_spatial_dimensions(3);
  dimension_numbers.set_kernel_output_feature_dimension(0);
  dimension_numbers.set_kernel_input_feature_dimension(1);
  dimension_numbers.add_kernel_spatial_dimensions(2);
  dimension_numbers.add_kernel_spatial_dimensions(3);
  dimension_numbers.set_output_batch_dimension(0);
  dimension_numbers.set_output_feature_dimension(1);
  dimension_numbers.add_output_spatial_dimensions(2);
  dimension_numbers.add_output_spatial_dimensions(3);

  xla::XlaOp result = xla::ConvGeneralDilated(inp,
                                              kernel,
                                              /*strides=*/{3, 3},
                                              /*padding=*/{},
                                              /*lhs_dilation=*/{},
                                              /*rhs_dilation=*/{},
                                              /*conv_dimnos=*/dimension_numbers);

  xla::XlaComputation comp = builder->Build(result).ConsumeValueOrDie();

  xla::ExecutableBuildOptions build_options;
  build_options.set_device_allocator(allocator.get());
  build_options.set_num_replicas(1);
  build_options.set_num_partitions(1);
  build_options.set_device_ordinal(1);

  std::vector<std::unique_ptr<xla::LocalExecutable>> exec =
    client->Compile(comp, {}, build_options).ConsumeValueOrDie();

  xla::ExecutableRunOptions run_options;

  run_options.set_allocator(allocator.get());
  run_options.set_device_ordinal(1);

  xla::ExecutionOutput out =
    exec.at(0)->Run(std::vector<xla::ExecutionInput>(), run_options).ConsumeValueOrDie();

  return enif_make_atom(env, "ok");
}

static ErlNifFunc conv_funcs[] = {
  {"conv", 0, conv},
  {"conv_seg_fault", 0, conv, ERL_NIF_DIRTY_JOB_IO_BOUND}
};

ERL_NIF_INIT(Elixir.ConvSegFault, conv_funcs, &load, NULL, NULL, NULL);