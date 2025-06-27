package compute_graph

import (
	"fmt"
	"testing"
)

func TestComputeGraph(t *testing.T) {
	graph := NewComputationalGraph()

	x := NewVariable("x", 2.0)
	w := NewVariable("w", 3.0)
	b := NewVariable("b", 1.0)

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

	learningRate := 0.1
	w.SetValue(w.value - learningRate*w.GetGrad())
	b.SetValue(b.value - learningRate*b.GetGrad())

	fmt.Println("\nAfter Parameter Update:")
	fmt.Printf("w updated: %.4f -> %.4f\n", 3.0, w.value)
	fmt.Printf("b updated: %.4f -> %.4f\n", 1.0, b.value)

	graph.Forward()
	fmt.Println("\nAfter Update Forward Pass:")
	fmt.Printf("Output: %.6f\n", sig.GetOutput())
}
