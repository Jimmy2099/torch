package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"testing"
)

func TestComputeGraph(t *testing.T) {
	graph := NewComputationalGraph()

	x := NewVariable("x", tensor.NewTensor([]float32{2.0, 2.0, 2.0, 2.0}, []int{2, 2}))
	w := NewVariable("w", tensor.NewTensor([]float32{3.0, 3.0, 3.0, 3.0}, []int{2, 2}))
	b := NewVariable("b", tensor.NewTensor([]float32{1.0, 1.0, 1.0, 1.0}, []int{2, 2}))

	graph.AddNode(x)
	graph.AddNode(w)
	graph.AddNode(b)

	mul := NewMultiply("w*x", w, x)
	add := NewAdd("wx+b", mul, b)
	sig := NewSigmoid("sigmoid", add)

	graph.AddNode(mul)
	graph.AddNode(add)
	graph.AddNode(sig)

	graph.Forward()
	fmt.Println("After Forward Pass:")
	graph.Print()

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	graph.Print()

	learningRate := tensor.NewTensor([]float32{0.1}, []int{1})
	w.SetValue(w.value.Sub(w.GetGrad().Mul(learningRate)))
	b.SetValue(b.value.Sub(b.GetGrad().Mul(learningRate)))
	//w.SetValue(w.value - learningRate*w.GetGrad())
	//b.SetValue(b.value - learningRate*b.GetGrad())

	fmt.Println("\nAfter Parameter Update:")
	fmt.Printf("w updated: %v -> %v\n", 3.0, w.value)
	fmt.Printf("b updated: %v -> %v\n", 1.0, b.value)

	graph.Forward()
	fmt.Println("\nAfter Update Forward Pass:")
	fmt.Printf("Output: %v\n", sig.GetOutput())
}
