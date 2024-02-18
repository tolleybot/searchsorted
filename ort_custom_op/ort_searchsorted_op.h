

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
    void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info)  {
        return new SearchSortedKernel<float>(api, info);
    }

    const char* GetName() const  { return "mydomain::searchsorted" }

    size_t GetInputTypeCount() const  { return 3; }; // Adjust based on your actual inputs
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const  { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

    size_t GetOutputTypeCount() const  { return 1; };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const  { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; }; // Adjust if needed
};

