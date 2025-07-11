package compute_graph

import (
	"fmt"
	"testing"
)

func TestNot(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensor
	a := graph.NewGraphTensor([]float32{1.0, 0.0, 0.5, 0.7}, []int{2, 2}, "a")

	// Create computation graph: result = NOT a
	result := a.Not("not_result")

	graph.SetOutput(result)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Result: %v\n", result.value.Data)

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("A Gradients: %v\n", a.Grad().Data)
}
