#include "onnxruntime_cxx_api.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

#define A_ROWS 30
#define A_COLS 1
#define V_ROWS 30
#define V_COLS 1
#define INT_DIST TRUE

/** Test function
 *
 */
template <typename T>
std::vector<int> searchsorted(const std::vector<T> &sorted_sequence,
                              const std::vector<T> &values, bool side_left) {
  std::vector<int> indices(values.size());

  for (size_t i = 0; i < values.size(); ++i) {
    auto it = side_left ? std::lower_bound(sorted_sequence.begin(),
                                           sorted_sequence.end(), values[i])
                        : std::upper_bound(sorted_sequence.begin(),
                                           sorted_sequence.end(), values[i]);
    indices[i] = std::distance(sorted_sequence.begin(), it);
  }

  return indices;
}

// Example validation logic
bool validateOutput(const std::vector<float> &a, const std::vector<float> &v,
                    const int64_t *output, const std::array<int64_t, 2> &dims_a,
                    const std::array<int64_t, 2> &dims_v,
                    const int64_t *output_dims) {
  // Assuming output is expected to be the index in 'a' for each value in 'v'
  for (int64_t i = 0; i < dims_v[0]; ++i) {   // Iterate over rows
    for (int64_t j = 0; j < dims_v[1]; ++j) { // Iterate over columns
      int64_t idx = i * dims_v[1] + j;        // Index in flat array
      int64_t res_idx = output[idx];          // Result index from output_data
      if (res_idx < 0 || res_idx >= dims_a[1]) { // Check index bounds
        std::cerr << "Invalid index " << res_idx << " at position (" << i
                  << ", " << j << ")\n";
        return false;
      }
    }
  }
  return true;
}

int main() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");

  Ort::SessionOptions session_options;
  OrtCUDAProviderOptions cuda_options;

  // session_options.SetInterOpNumThreads(1);
  // session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_DISABLE_ALL);

  cuda_options.device_id = 0; // GPU_ID
  cuda_options.cudnn_conv_algo_search =
      OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
  cuda_options.arena_extend_strategy = 0;
  // May cause data race in some condition
  cuda_options.do_copy_in_default_stream = 0;
  session_options.AppendExecutionProvider_CUDA(
      cuda_options); // Add CUDA options to session options

  // Assuming libortsearchsortedop.so is in the current working directory's
  // "build" subdirectory
  std::string library_path = "/src/build/libortsearchsortedop.so";
  session_options.RegisterCustomOpsLibrary(library_path.c_str());
  session_options.SetLogSeverityLevel(0); // Enable more verbose logging

  const char *model_path = "/src/searchsorted_custom_op.onnx";

  // Use CUDAExecutionProvider if available, otherwise default to CPU , ERROR
  // happens on this line
  Ort::Session session(env, model_path, session_options);

  // Prepare input tensors
  std::vector<float> a_dummy(A_ROWS *
                             A_COLS); // Adjust size based on your model's input
  std::vector<float> v_dummy(V_ROWS * V_COLS);
  std::vector<uint8_t> side_left_data = {
      1}; // Representing 'true' as 1 (use 0 for 'false')

  std::random_device rd;
  std::mt19937 gen(rd());

#if INT_DIST == TRUE
  std::uniform_int_distribution<> dis(0, 100);
#else
std:
  uniform_real_distribution<float> dis(0.0f, 1.0f);
#endif

  std::cout << "-----------------------------" << std::endl;
  std::cout << "A" << std::endl;
  std::cout << "-----------------------------" << std::endl;
  for (auto &value : a_dummy) {
    value = dis(gen);
  }
  std::sort(a_dummy.begin(), a_dummy.end());
  for (auto &value : a_dummy) {
    std::cout << value << std::endl;
  }

  std::cout << "-----------------------------" << std::endl;
  std::cout << "V" << std::endl;
  std::cout << "-----------------------------" << std::endl;
  for (auto &value : v_dummy) {
    value = dis(gen);
    std::cout << value << std::endl;
  }
  std::cout << "-----------------------------" << std::endl;
  std::cout << "Expected Results" << std::endl;
  std::cout << "-----------------------------" << std::endl;
  std::vector<int> expected_indices =
      searchsorted<float>(a_dummy, v_dummy, true);
  for (auto &value : expected_indices) {
    std::cout << value << std::endl;
  }

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value a_tensor = Ort::Value::CreateTensor<float>(
      memory_info, a_dummy.data(), a_dummy.size(),
      (std::array<int64_t, 2>{A_ROWS, A_COLS}).data(), 2);
  Ort::Value v_tensor = Ort::Value::CreateTensor<float>(
      memory_info, v_dummy.data(), v_dummy.size(),
      (std::array<int64_t, 2>{V_ROWS, V_COLS}).data(), 2);
  Ort::Value side_left_tensor = Ort::Value::CreateTensor<bool>(
      memory_info,
      reinterpret_cast<bool *>(side_left_data.data()), // Cast uint8_t* to bool*
      side_left_data.size(), (std::array<int64_t, 1>{1}).data(),
      1); // Shape for a single boolean value

  // Run the model
  const char *input_names[] = {"a", "v", "side_left"};
  Ort::Value input_tensors[] = {std::move(a_tensor), std::move(v_tensor),
                                std::move(side_left_tensor)};
  const char *output_names[] = {"output"};

  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                             input_tensors, 3, output_names, 1);

  // Process output tensors
  // Assuming output is a tensor of floats
  int64_t *output_data = outputs.front().GetTensorMutableData<int64_t>();
  // Output tensor dimensions
  Ort::TensorTypeAndShapeInfo output_info =
      outputs.front().GetTensorTypeAndShapeInfo();
  auto output_dims = output_info.GetShape();

  bool isValid = validateOutput(a_dummy, v_dummy, output_data, {A_ROWS, A_COLS},
                                {V_ROWS, V_COLS}, output_dims.data());
  if (!isValid) {
    std::cerr << "Output validation failed.\n";
  } else {
    std::cout << "Output validation succeeded.\n";
  }

  // Process the output as needed, for example, print the first 10 values
  for (int i = 0; i < 10 && i < output_dims[0]; ++i) {
    std::cout << "Output value " << i << ": " << output_data[i] << std::endl;
  }

  std::cout << "Finished running the model" << std::endl;

  return 0;
}
