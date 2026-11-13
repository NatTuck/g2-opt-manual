/*******************************************************************************
  hpu_custom_add1: HPU custom op that adds 1 to every element of a tensor.
  Calls TPC kernel GUID "add1_fwd_f32_gaudi2".
 *******************************************************************************/

#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"
#include <perf_lib_layer_params.h>

// Callback describing output tensors parameters.
static habana::PartialOutputMetaDataVector output_meta(
    const at::Stack& inputs) {
  habana::PartialOutputMetaData meta_output{
      c10::ScalarType::Float, inputs[0].toTensor().sizes().vec()};
  return {meta_output};
}

// Callback describing TPC kernel parameters.
// The add1 TPC kernel has no scalar params (the +1 is hardcoded inside the
// kernel), but the descriptor still requires a params stub to be returned.
struct Add1Params {
  float dummy;
};

static std::shared_ptr<void> fill_params(
    const at::Stack& inputs,
    size_t& size) {
  HPU_PARAMS_STUB(Add1Params);
    params->dummy = 0.0f;
    return params;
}

// Registration of custom_add1 in Habana registry.
bool register_custom_add1() {
  habana::custom_op::registerUserCustomOp(
      "custom_op::custom_add1", "add1_fwd_f32_gaudi2", output_meta, fill_params);
  return true;
}

// Call to the registered operator.
at::Tensor custom_add1(at::Tensor input_a) {
  TORCH_CHECK(input_a.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");

  // Get custom op descriptor from Habana registry.
  auto op_desc =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          "custom_op::custom_add1");
  std::vector<c10::IValue> inputs{input_a};
  // Actual call for op execution.
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // Output is always a vector of at::Tensor.
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_add1(Tensor self) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_add1", custom_add1);
}

// Meta kernel implementation describing output tensors.
// Needed only for torch.compile support.
at::Tensor custom_add1_meta(at::Tensor input_a) {
  return input_a.new_empty(input_a.sizes().vec(), c10::ScalarType::Float);
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("custom_add1", custom_add1_meta);
}

// Registration of custom_add1 in Habana registry.
static const auto& KernelReg = register_custom_add1();
