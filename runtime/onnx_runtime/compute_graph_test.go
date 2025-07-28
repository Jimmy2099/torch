package onnx_runtime

import (
	"bufio"
	"fmt"
	"github.com/Jimmy2099/torch/data_store/compute_graph"
	test "github.com/Jimmy2099/torch/testing"
	"os"
	"testing"
)

func genPyScriptInP1OutP1(opsName string) string {
	return fmt.Sprintf(`
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
}

func genPyScriptInP2OutP1(opsName string) string {
	return fmt.Sprintf(`
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

# Create Add model
model = helper.make_model(
    helper.make_graph(
        [helper.make_node("%s", ["input1", "input2"], ["output"])],
        "test_graph",
        [
            helper.make_tensor_value_info("input1", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("input2", TensorProto.FLOAT, [4])
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])]
    ),
    opset_imports=[helper.make_opsetid("", 14)],  # Current opset
    ir_version=8  # Current IR version
)

# Verify model
onnx.checker.check_model(model)
print(f"Model created with IR version: {model.ir_version}, Opset: {model.opset_import[0].version}")

# Load model
session = ort.InferenceSession(model.SerializeToString())

# Test inference
input1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
input2 = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

results = session.run(None, {"input1": input1, "input2": input2})[0]

print("Input1:", input1)
print("Input2:", input2)
print("Output:", results)
`, opsName)
}

func TestONNX(t *testing.T) {
	opsNameList := compute_graph.ONNXOperators
	for i := 0; i < len(opsNameList); i++ {
		if opsNameList[i].Ignore == false &&
			opsNameList[i].InputPCount == 1 &&
			opsNameList[i].OutputPCount == 1 {
			pythonScript := genPyScriptInP1OutP1(opsNameList[i].Name)
			test.RunPyScript(pythonScript)
		}
		if opsNameList[i].Ignore == false &&
			opsNameList[i].InputPCount == 2 &&
			opsNameList[i].OutputPCount == 1 {
			pythonScript := genPyScriptInP2OutP1(opsNameList[i].Name)
			test.RunPyScript(pythonScript)
		}
	}
}

func writeLog(content string) {
	file, err := os.OpenFile("result.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer file.Close()

	writer := bufio.NewWriter(file)

	if _, err = writer.WriteString(content); err != nil {
		return
	}

	if err = writer.Flush(); err != nil {
		return
	}
}
