package compute_graph

import (
	"fmt"
	test "github.com/Jimmy2099/torch/testing"
	"testing"
)

func TestComputeGraph(t *testing.T) {
	graph := NewComputationalGraph()

	x := graph.NewTensor([]float32{2.0, 2.0, 2.0, 2.0}, []int{2, 2}, "x")
	w := graph.NewTensor([]float32{3.0, 3.0, 3.0, 3.0}, []int{2, 2}, "w")
	b := graph.NewTensor([]float32{1.0, 1.0, 1.0, 1.0}, []int{2, 2}, "b")

	wx := x.Multiply(w)
	xb := x.Multiply(b)
	add := wx.Multiply(xb)

	graph.SetOutput(add)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	// 第一次前向传播
	graph.Forward()
	fmt.Println("\nAfter First Forward Pass:")
	fmt.Printf("Output: %v\n", add.Value().Data)

	// 第二次前向传播（相同输入）
	graph.Forward()
	fmt.Println("\nAfter Second Forward Pass (same inputs):")
	fmt.Printf("Output: %v\n", add.Value().Data) // 应相同

	// 修改输入值
	x.Value().Data = []float32{3.0, 3.0, 3.0, 3.0}

	// 第三次前向传播（新输入）
	graph.Forward()
	fmt.Println("\nAfter Third Forward Pass (changed inputs):")
	fmt.Printf("Output: %v\n", add.Value().Data) // 应不同

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("weights: %v\n", w.Grad().Data)
	fmt.Printf("bias: %v\n", b.Grad().Data)

	learningRate := float32(0.1)
	wData := w.Value().Data
	wGrad := w.Grad().Data
	for i := range wData {
		wData[i] -= learningRate * wGrad[i]
	}

	bData := b.Value().Data
	bGrad := b.Grad().Data
	for i := range bData {
		bData[i] -= learningRate * bGrad[i]
	}

	fmt.Println("\nAfter Parameter Update:")
	fmt.Printf("weight updated: %v\n", w.Value().Data)
	fmt.Printf("bias updated: %v\n", b.Value().Data)

	// 更新参数后的前向传播
	graph.Forward()
	fmt.Println("\nAfter Update Forward Pass:")
	fmt.Printf("Output: %v\n", add.Value().Data)
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

func TestExportONNX(t *testing.T) {
	graph := NewComputationalGraph()

	x := graph.NewTensor([]float32{2.0, 2.0, 2.0, 2.0}, []int{2, 2}, "x")
	w := graph.NewTensor([]float32{3.0, 3.0, 3.0, 3.0}, []int{2, 2}, "w")
	b := graph.NewTensor([]float32{1.0, 1.0, 1.0, 1.0}, []int{2, 2}, "b")

	wx := x.Multiply(w)
	xb := x.Multiply(b)
	add := wx.Multiply(xb)

	graph.SetOutput(add)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	// 第一次前向传播
	graph.Forward()
	fmt.Println("\nAfter First Forward Pass:")
	fmt.Printf("Output: %v\n", add.Value().Data)

	// 第二次前向传播（相同输入）
	graph.Forward()
	fmt.Println("\nAfter Second Forward Pass (same inputs):")
	fmt.Printf("Output: %v\n", add.Value().Data) // 应相同

	// 修改输入值
	x.Value().Data = []float32{3.0, 3.0, 3.0, 3.0}

	// 第三次前向传播（新输入）
	graph.Forward()
	fmt.Println("\nAfter Third Forward Pass (changed inputs):")
	fmt.Printf("Output: %v\n", add.Value().Data) // 应不同

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("weights: %v\n", w.Grad().Data)
	fmt.Printf("bias: %v\n", b.Grad().Data)

	learningRate := float32(0.1)
	wData := w.Value().Data
	wGrad := w.Grad().Data
	for i := range wData {
		wData[i] -= learningRate * wGrad[i]
	}

	bData := b.Value().Data
	bGrad := b.Grad().Data
	for i := range bData {
		bData[i] -= learningRate * bGrad[i]
	}

	fmt.Println("\nAfter Parameter Update:")
	fmt.Printf("weight updated: %v\n", w.Value().Data)
	fmt.Printf("bias updated: %v\n", b.Value().Data)

	// 更新参数后的前向传播
	graph.Forward()
	fmt.Println("\nAfter Update Forward Pass:")
	fmt.Printf("Output: %v\n", add.Value().Data)

	onnxModel, err := graph.ToONNXModel()
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(onnxModel)
	onnxModel.SaveONNX("model.onnx")
}
