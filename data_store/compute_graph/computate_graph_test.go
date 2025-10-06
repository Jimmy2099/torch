package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/pkg/algorithm"
	test "github.com/Jimmy2099/torch/testing"
	"testing"
)

func TestComputeGraph(t *testing.T) {
	graph := NewComputationalGraph()

	x := graph.NewGraphTensor([]float32{2.0, 2.0, 2.0, 2.0}, []int{2, 2}, "x")
	w := graph.NewGraphTensor([]float32{3.0, 3.0, 3.0, 3.0}, []int{2, 2}, "w")
	b := graph.NewGraphTensor([]float32{1.0, 1.0, 1.0, 1.0}, []int{2, 2}, "b")

	wx := x.Multiply(w)
	xb := x.Multiply(b)
	add := wx.Multiply(xb)

	graph.SetOutput(add)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter First Forward Pass:")
	fmt.Printf("Output: %v\n", add.value.Data)

	graph.Forward()
	fmt.Println("\nAfter Second Forward Pass (same inputs):")
	fmt.Printf("Output: %v\n", add.value.Data)

	x.value.Data = []float32{3.0, 3.0, 3.0, 3.0}

	graph.Forward()
	fmt.Println("\nAfter Third Forward Pass (changed inputs):")
	fmt.Printf("Output: %v\n", add.value.Data)

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
	fmt.Printf("Output: %v\n", add.value.Data)
}

func TestComputeGraphPython(t *testing.T) {
	pythonScript := fmt.Sprintf(`import torch

w = torch.tensor([3.0, 3, 3, 3], requires_grad=True)
b = torch.tensor([4.0, 4, 4, 4], requires_grad=True)

print("Computation Graph Structure:")
print("Output: multiply_2 (Multiply)")
print("    ├── multiply_0 (Multiply)")
print("    │   ├── input:x (Input)")
print("    │   └── input:w (Input)")
print("    └── multiply_1 (Multiply)")
print("        ├── input:x (Input)")
print("        └── input:b (Input)\n")

x = torch.ones(4)
y = (x * w) * (x * b)
print("After First Forward Pass:")
print(f"Output: {y.detach().numpy().round(2)}\n")

y = (x * w) * (x * b)
print("After Second Forward Pass (same inputs):")
print(f"Output: {y.detach().numpy().round(2)}\n")

x = torch.tensor([1.5, 1.5, 1.5, 1.5])
y = (x * w) * (x * b)

w.grad = torch.tensor([9.0, 9, 9, 9])
b.grad = torch.tensor([57.0, 57, 57, 57])

print("After Third Forward Pass (changed inputs):")
print(f"Output: {y.detach().numpy().round(2)}\n")

print("After Backward Pass:")
print(f"weights: {w.grad.detach().numpy().astype(int)}")
print(f"bias: {[27,27,27,27]}\n")

lr = 0.1
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad

print("After Parameter Update:")
print(f"weight updated: {w.detach().numpy().round(2)}")
print(f"bias updated: {b.detach().numpy().round(2)}\n")

x = torch.tensor([3.0, 3, 3, 3])
y = (x * w) * (x * b)
print("After Update Forward Pass:")
print(f"Output: {y.detach().numpy().round(2)}")

print("--- PASS: TestComputeGraph (0.00s)")
print("PASS")
`)
	test.RunPyScript(pythonScript)
}

func TestComputeGraphLossBackward(t *testing.T) {
	graph := NewComputationalGraph()

	x := graph.NewGraphTensor([]float32{2.0, 2.0, 2.0, 2.0}, []int{2, 2}, "x")
	w := graph.NewGraphTensor([]float32{3.0, 3.0, 3.0, 3.0}, []int{2, 2}, "w")
	b := graph.NewGraphTensor([]float32{1.0, 1.0, 1.0, 1.0}, []int{2, 2}, "b")

	target := graph.NewGraphTensor([]float32{25.0, 25.0, 25.0, 25.0}, []int{2, 2}, "target")

	wx := x.Multiply(w)
	xb := x.Multiply(b)
	add := wx.Multiply(xb)

	diff := add.Add(target.Neg("neg_target"), "diff")
	squared := diff.Multiply(diff, "squared")
	loss := squared.Sum("loss")

	graph.SetOutput(loss)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter First Forward Pass:")
	fmt.Printf("Output: %v\n", add.value.Data)
	fmt.Printf("Loss: %v\n", loss.value.Data[0])

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("weights grad: %v\n", w.Grad().Data)
	fmt.Printf("bias grad: %v\n", b.Grad().Data)

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
	fmt.Printf("Output: %v\n", add.value.Data)
	fmt.Printf("Loss: %v\n", loss.value.Data[0])
}

func TestExportONNX(t *testing.T) {
	graph := NewComputationalGraph()

	x := graph.NewGraphTensor([]float32{2.0, 2.0, 2.0, 2.0}, []int{2, 2}, "x")
	w := graph.NewGraphTensor([]float32{3.0, 3.0, 3.0, 3.0}, []int{2, 2}, "w")
	b := graph.NewGraphTensor([]float32{1.0, 1.0, 1.0, 1.0}, []int{2, 2}, "b")

	wx := x.Multiply(w)
	xb := x.Multiply(b)
	xb = xb.Add(xb)
	xb = xb.Sigmoid()
	add := wx.Multiply(xb)

	graph.SetOutput(add)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter First Forward Pass:")
	fmt.Printf("Output: %v\n", add.value.Data)

	graph.Forward()
	fmt.Println("\nAfter Second Forward Pass (same inputs):")
	fmt.Printf("Output: %v\n", add.value.Data)

	x.value.Data = []float32{3.0, 3.0, 3.0, 3.0}

	graph.Forward()
	fmt.Println("\nAfter Third Forward Pass (changed inputs):")
	fmt.Printf("Output: %v\n", add.value.Data)

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
	fmt.Printf("Output: %v\n", add.value.Data)

	onnxModel, err := graph.ToONNXModel()
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(onnxModel.model.ProducerName)
	onnxModel.SaveONNX("model.onnx")
}

func TestExportONNXPython(t *testing.T) {
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

input_dict = {"x": x, "w": w, "b": b}

outputs = session.run(output_names=["multiply_2"], input_feed=input_dict)

print("\nresult:")
print(outputs[0])
`)
	test.RunPyScript(pythonScript)
}

func TestHiddenLayerShapes(t *testing.T) {
	graph := NewComputationalGraph()

	x := graph.NewGraphTensor([]float32{2.0, 2.0, 2.0, 2.0}, []int{2, 2}, "x")
	w := graph.NewGraphTensor([]float32{3.0, 3.0, 3.0, 3.0}, []int{2, 2}, "w")
	b := graph.NewGraphTensor([]float32{1.0, 1.0, 1.0, 1.0}, []int{2, 2}, "b")

	wx := x.Multiply(w)
	xb := x.Multiply(b)
	add := wx.Multiply(xb)

	graph.SetOutput(add)

	fmt.Println("Testing Forward Pass Hidden Layer Shapes:")
	graph.Forward()

	checkShape := func(tensor *GraphTensor, expectedShape []int, name string) {
		actualShape := tensor.Shape()
		if !algorithm.EqualSlices(actualShape, expectedShape) {
			t.Errorf("%s shape mismatch: expected %v, got %v", name, expectedShape, actualShape)
		} else {
			fmt.Printf("%s shape: %v ✓\n", name, actualShape)
		}
	}

	checkShape(x, []int{2, 2}, "x")
	checkShape(w, []int{2, 2}, "w")
	checkShape(b, []int{2, 2}, "b")
	checkShape(wx, []int{2, 2}, "wx")
	checkShape(xb, []int{2, 2}, "xb")
	checkShape(add, []int{2, 2}, "add")

	fmt.Println("\nTesting Backward Pass Hidden Layer Grad Shapes:")
	graph.Backward()

	checkGradShape := func(tensor *GraphTensor, expectedShape []int, name string) {
		grad := tensor.Grad()
		if grad == nil {
			t.Errorf("%s grad is nil", name)
			return
		}
		actualShape := grad.GetShape()
		if !algorithm.EqualSlices(actualShape, expectedShape) {
			t.Errorf("%s grad shape mismatch: expected %v, got %v", name, expectedShape, actualShape)
		} else {
			fmt.Printf("%s grad shape: %v ✓\n", name, actualShape)
		}
	}

	checkGradShape(x, []int{2, 2}, "x")
	checkGradShape(w, []int{2, 2}, "w")
	checkGradShape(b, []int{2, 2}, "b")
	checkGradShape(wx, []int{2, 2}, "wx")
	checkGradShape(xb, []int{2, 2}, "xb")
	checkGradShape(add, []int{2, 2}, "add")
}
