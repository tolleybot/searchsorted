

#pragma once

#include "core/session/onnxruntime_cxx_api.h"
#include "/onnxruntime/onnxruntime/core/session/custom_ops.h"
#include <vector>

// Helper to get tensor dimensions
struct OrtTensorDimensions : std::vector<int64_t> {
    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
        OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
        std::vector<int64_t>::operator=(ort.GetTensorShape(info));
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
};

template <typename T>
struct SearchSortedKernel {
private:
    Ort::CustomOpApi ort_;
    int64_t side_; // Assuming 'side' is a simple attribute controlling the behavior of searchsorted

public:
    SearchSortedKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info) : ort_(ort) {
        // Assuming 'side' is provided as an attribute. If it's an input tensor, you'll need to handle it differently
        side_ = ort_.KernelInfoGetAttribute<int64_t>(info, "side");
    }

    void Compute(OrtKernelContext* context);
};

struct SearchSortedCustomOp : Ort::CustomOpBase<SearchSortedCustomOp, SearchSortedKernel<float>> {
    void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info)  {
        return new SearchSortedKernel<float>(api, info);
    }

    const char* GetName() const  { return "SearchSorted"; }

    size_t GetInputTypeCount() const  { return 2; }; // Adjust based on your actual inputs
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const  { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

    size_t GetOutputTypeCount() const  { return 1; };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const  { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; }; // Adjust if needed
};

