package compute_graph

import (
	"fmt"
	"testing"
)

func TestBatchNormalization(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensors
	input := graph.NewGraphTensor([]float32{1.0, 2.0, 3.0, 4.0}, []int{2, 2}, "input")
	gamma := graph.NewGraphTensor([]float32{0.5, 0.5}, []int{2}, "gamma")
	beta := graph.NewGraphTensor([]float32{0.1, 0.1}, []int{2}, "beta")

	// Create batch normalization
	bn := input.BatchNormalization(gamma, beta, 1e-5, 0.9, "bn_output")
	graph.SetOutput(bn)

	fmt.Println("Batch Normalization Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("BN Output: %v\n", bn.value.Data)

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
	fmt.Printf("Gamma Gradients: %v\n", gamma.Grad().Data)
	fmt.Printf("Beta Gradients: %v\n", beta.Grad().Data)
}
