package compute_graph

import (
	"fmt"
	"testing"
)

func TestLRN(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{
		1.0, 2.0,
		3.0, 4.0,
		5.0, 6.0,
		7.0, 8.0,
	}, []int{2, 2, 2, 1}, "input")

	lrn := input.LRN(5, 0.0001, 0.75, 2.0, "lrn_output")
	graph.SetOutput(lrn)

	fmt.Println("LRN Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("LRN Output: %v\n", lrn.value.Data)

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
