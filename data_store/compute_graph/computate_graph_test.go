package compute_graph

import (
	"fmt"
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
	fmt.Printf("w updated: %v\n", w.Value().Data)
	fmt.Printf("b updated: %v\n", b.Value().Data)

	// 更新参数后的前向传播
	graph.Forward()
	fmt.Println("\nAfter Update Forward Pass:")
	fmt.Printf("Output: %v\n", add.Value().Data)
}
