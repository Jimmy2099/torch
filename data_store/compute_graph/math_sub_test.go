package compute_graph

import (
	"fmt"
	test "github.com/Jimmy2099/torch/testing"
	"testing"
)

func ONNXPythonLoadTest() {
	pythonScript := fmt.Sprintf(`import onnxruntime as ort
import numpy as np

model_path = "model.onnx"
session = ort.InferenceSession(model_path)

print("input:")
for i, input in enumerate(session.get_inputs()):
    print(f"{i}: name={input.name}, shape={input.shape}, type={input.type}")

for i, output in enumerate(session.get_outputs()):
    print(f"{i}: name={output.name}, shape={output.shape}, type={output.type}")

x = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32).reshape(2, 2)
w = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32).reshape(2, 2)
b = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(2, 2)
c = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32).reshape(2, 2)

input_dict = {"x": x, "w": w, "b": b, "c": c}

outputs = session.run(output_names=["sub_3"], input_feed=input_dict)

print("\nresult:")
print(outputs[0])
`)
	test.RunPyScript(pythonScript)
}

func TestSub(t *testing.T) {
	graph := NewComputationalGraph()

	x := graph.NewGraphTensor([]float32{2.0, 2.0, 2.0, 2.0}, []int{2, 2}, "x")
	w := graph.NewGraphTensor([]float32{3.0, 3.0, 3.0, 3.0}, []int{2, 2}, "w")
	b := graph.NewGraphTensor([]float32{1.0, 1.0, 1.0, 1.0}, []int{2, 2}, "b")

	c := graph.NewGraphTensor([]float32{2.0, 2.0, 2.0, 2.0}, []int{2, 2}, "c")

	wx := x.Multiply(w)
	xb := x.Multiply(b)
	result := wx.Multiply(xb).Sub(c)

	graph.SetOutput(result)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter First Forward Pass:")
	fmt.Printf("Output: %v\n", result.value.Data)

	graph.Forward()
	fmt.Println("\nAfter Second Forward Pass (same inputs):")
	fmt.Printf("Output: %v\n", result.value.Data)

	x.value.Data = []float32{3.0, 3.0, 3.0, 3.0}

	graph.Forward()
	fmt.Println("\nAfter Third Forward Pass (changed inputs):")
	fmt.Printf("Output: %v\n", result.value.Data)

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("weights: %v\n", w.Grad().Data)
	fmt.Printf("bias: %v\n", b.Grad().Data)

	learningRate := float32(0.1)
	wData := w.value.Data
	wGrad := w.Grad().Data
	for i := range wData {
		wData[i] -= learningRate * wGrad[i]
	}

	bData := b.value.Data
	bGrad := b.Grad().Data
	for i := range bData {
		bData[i] -= learningRate * bGrad[i]
	}

	fmt.Println("\nAfter Parameter Update:")
	fmt.Printf("weight updated: %v\n", w.value.Data)
	fmt.Printf("bias updated: %v\n", b.value.Data)

	graph.Forward()
	fmt.Println("\nAfter Update Forward Pass:")
	fmt.Printf("Output: %v\n", result.value.Data)

	onnxModel, err := graph.ToONNXModel()
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(onnxModel.model.ProducerName)
	onnxModel.SaveONNX("model.onnx")
	ONNXPythonLoadTest()
}
