package onnx_runtime

import (
	"fmt"
	test "github.com/Jimmy2099/torch/testing"
	"testing"
)

func TestONNX(t *testing.T) {
	pythonScript := fmt.Sprintf(`
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

# Create ABS model with compatible settings
model = helper.make_model(
    helper.make_graph(
        [helper.make_node("Abs", ["input"], ["output"])],
        "abs-test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])]
    ),
    opset_imports=[helper.make_opsetid("", 10)],  # Reduced opset
    ir_version=7  # Explicitly set compatible IR version
)

# Verify model
onnx.checker.check_model(model)
print(f"Model created with IR version: {model.ir_version}, Opset: {model.opset_import[0].version}")

# Load model
session = ort.InferenceSession(model.SerializeToString())

# Test inference
input_data = np.array([-1.0, 0.0, 2.5, -3.5], dtype=np.float32)
results = session.run(None, {"input": input_data})[0]

print("Input:", input_data)
print("Output:", results)
`)
	test.RunPyScript(pythonScript)
}
