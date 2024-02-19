

#pragma once

#include "onnxruntime_cxx_api.h"
#include "/onnxruntime/onnxruntime/core/session/custom_ops.h"
#include <vector>

// Helper to get tensor dimensions
struct OrtTensorDimensions : std::vector<int64_t> {
    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
        OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
        size_t dim_count = ort. GetDimensionsCount(info);
        this->resize(dim_count);
        ort.GetDimensions(info, this->data(), dim_count);
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
};



template <typename T>
struct SearchSortedKernel {
private:
    Ort::CustomOpApi ort_;
public:
    SearchSortedKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info) : ort_(ort) {

    }

    void Compute(OrtKernelContext* context);
};

struct SearchSortedCustomOp : Ort::CustomOpBase<SearchSortedCustomOp, SearchSortedKernel<float>> {
   // Assuming CreateKernel matches the expected signature from the CustomOpBase setup
    void* CreateKernel(const Ort::CustomOpApi api, const OrtKernelInfo* info) const {
        // Pass the API and info to your kernel's constructor, if needed
        return new SearchSortedKernel<float>(api, info);
    }

    const char* GetName() const { return "searchsorted"; }

    size_t GetInputTypeCount() const { return 3; } // Adjust based on your actual inputs
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

    size_t GetOutputTypeCount() const { return 1; };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; }; // Adjust if needed

    // If your custom op supports a specific execution provider (e.g., CUDA), implement this:
    // Note: Remove or adjust this method based on whether your op supports specific providers
    const char* GetExecutionProviderType() const  {
        // Return nullptr or the specific provider type your op supports
        // For CPU, you could simply return nullptr to indicate no specific provider requirement
        // For CUDA, return "CUDAExecutionProvider";
        #ifdef USE_CUDA
        return "CUDAExecutionProvider";
        #else
        return nullptr;
        #endif

    }
};

