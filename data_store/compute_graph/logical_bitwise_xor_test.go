package compute_graph

import (
	"fmt"
	"testing"
)

func TestXor(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensors
	a := graph.NewGraphTensor([]float32{1.0, 0.0, 1.0, 0.0}, []int{2, 2}, "a")
	b := graph.NewGraphTensor([]float32{1.0, 1.0, 0.0, 0.0}, []int{2, 2}, "b")

	// Create computation graph: result = a XOR b
	result := a.Xor(b, "xor_result")

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
	fmt.Printf("B Gradients: %v\n", b.Grad().Data)
}
