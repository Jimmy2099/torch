package compute_graph

import (
	"fmt"
	"testing"
)

func TestAbs(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{-2.0, 3.0, -4.0, 0.0}, []int{2, 2}, "input")
	abs := input.Abs("abs_result")

	graph.SetOutput(abs)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Abs: %v\n", abs.value.Data)

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
