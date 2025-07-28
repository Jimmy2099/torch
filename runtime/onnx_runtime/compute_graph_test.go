package onnx_runtime

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/compute_graph"
	test "github.com/Jimmy2099/torch/testing"
	"testing"
)

func genPyScriptInP1OutP1(opsName string) string {
	pythonScript := fmt.Sprintf(`
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

# Create ABS model with compatible settings
model = helper.make_model(
    helper.make_graph(
        [helper.make_node("%s", ["input"], ["output"])],
        "test",
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
`, opsName)

	return pythonScript
}

func TestONNX(t *testing.T) {
	opsNameList := compute_graph.ONNXOperators
	for i := 0; i < len(opsNameList); i++ {
		fmt.Println(opsNameList[i])
		if opsNameList[i].InputPCount == 1 && opsNameList[i].OutputPCount == 1 {
			pythonScript := genPyScriptInP1OutP1(opsNameList[i].Name)
			test.RunPyScript(pythonScript)
		}
	}
}
