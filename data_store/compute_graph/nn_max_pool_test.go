package compute_graph

import (
	"fmt"
	"testing"
)

func TestMaxPool(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensor: 1 batch, 1 channel, 4x4
	input := graph.NewGraphTensor(
		[]float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		},
		[]int{1, 1, 4, 4},
		"input",
	)

	// Create max pooling layer
	pooled := input.MaxPool(
		[]int{2, 2}, // kernel
		[]int{2, 2}, // stride
		[]int{0, 0}, // padding
		"max_pool",
	)

	graph.SetOutput(pooled)

	fmt.Println("MaxPool Computation Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Pooled Output: %v\n", pooled.value.Data)

	// Expected: [max(1,2,5,6), max(3,4,7,8), max(9,10,13,14), max(11,12,15,16)]
	expected := []float32{6, 8, 14, 16}
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
		0, 0, 0, 0,
		0, 1, 0, 1,
		0, 0, 0, 0,
		0, 1, 0, 1,
	}
	for i, grad := range input.Grad().Data {
		if grad != expectedGrad[i] {
			t.Errorf("At index %d, expected gradient %.2f, got %.2f", i, expectedGrad[i], grad)
		}
	}
}
