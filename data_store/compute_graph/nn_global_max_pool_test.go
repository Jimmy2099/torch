package compute_graph

import (
	"fmt"
	"testing"
)

func TestGlobalMaxPool(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensor: 1 batch, 2 channels, 2x2
	input := graph.NewGraphTensor(
		[]float32{
			1, 2, // Channel 1
			3, 4,
			5, 6, // Channel 2
			7, 8,
		},
		[]int{1, 2, 2, 2},
		"input",
	)

	// Create global max pooling layer
	pooled := input.GlobalMaxPool("global_max_pool")

	graph.SetOutput(pooled)

	fmt.Println("GlobalMaxPool Computation Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Pooled Output: %v\n", pooled.value.Data)

	// Expected: [max(1,2,3,4), max(5,6,7,8)]
	expected := []float32{4, 8}
	if len(pooled.value.Data) != len(expected) {
		t.Errorf("Expected %d outputs, got %d", len(expected), len(pooled.value.Data))
	}
	for i, val := range pooled.value.Data {
		if val != expected[i] {
			t.Errorf("At index %d, expected %.2f, got %.2f", i, expected[i], val)
		}
	}

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)

	// Expected gradients: only max positions receive gradient
	expectedGrad := []float32{
		0, 0, // Channel 1
		0, 1,
		0, 0, // Channel 2
		0, 1,
	}
	for i, grad := range input.Grad().Data {
		if grad != expectedGrad[i] {
			t.Errorf("At index %d, expected gradient %.2f, got %.2f", i, expectedGrad[i], grad)
		}
	}
}
