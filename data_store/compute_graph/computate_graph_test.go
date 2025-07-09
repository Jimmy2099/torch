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

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Output: %v\n", add.Value().Data)

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("w gradient: %v\n", w.Grad().Data)
	fmt.Printf("b gradient: %v\n", b.Grad().Data)

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

	graph.Forward()
	fmt.Println("\nAfter Update Forward Pass:")
	fmt.Printf("Output: %v\n", add.Value().Data)
}
