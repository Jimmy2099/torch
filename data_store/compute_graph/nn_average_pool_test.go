package compute_graph

import (
	"fmt"
	"testing"
)

func TestAveragePool(t *testing.T) {
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

	// Create average pooling layer
	pooled := input.AveragePool(
		[]int{2, 2}, // kernel
		[]int{2, 2}, // stride
		[]int{0, 0}, // padding
		"avg_pool",
	)

	graph.SetOutput(pooled)

	fmt.Println("AveragePool Computation Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Pooled Output: %v\n", pooled.value.Data)

	// Expected: [(1+2+5+6)/4, (3+4+7+8)/4, (9+10+13+14)/4, (11+12+15+16)/4]
	expected := []float32{3.5, 5.5, 11.5, 13.5}
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

	// Expected gradients: each element in kernel receives 1/4 of gradient
	expectedGrad := []float32{
		0.25, 0.25, 0.25, 0.25,
		0.25, 0.25, 0.25, 0.25,
		0.25, 0.25, 0.25, 0.25,
		0.25, 0.25, 0.25, 0.25,
	}
	for i, grad := range input.Grad().Data {
		if grad != expectedGrad[i] {
			t.Errorf("At index %d, expected gradient %.4f, got %.4f", i, expectedGrad[i], grad)
		}
	}
}
