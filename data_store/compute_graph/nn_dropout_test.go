package compute_graph

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestDropout(t *testing.T) {
	graph := NewComputationalGraph()
	rand.Seed(42) // For reproducible tests

	input := graph.NewGraphTensor([]float32{1.0, 2.0, 3.0, 4.0}, []int{2, 2}, "input")
	dropout := input.Dropout(0.5, "dropout_output")
	graph.SetOutput(dropout)

	fmt.Println("Dropout Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Dropout Output: %v\n", dropout.value.Data)

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
