package compute_graph

import (
	"fmt"
	"testing"
)

func TestDiv(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensors
	numerator := graph.NewGraphTensor([]float32{8.0, 12.0, 16.0, 20.0}, []int{2, 2}, "num")
	denominator := graph.NewGraphTensor([]float32{2.0, 3.0, 4.0, 5.0}, []int{2, 2}, "denom")

	// Create computation graph: result = num / denom
	quotient := numerator.Div(denominator, "division_result")

	graph.SetOutput(quotient)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Quotient: %v\n", quotient.value.Data)

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Numerator Gradients: %v\n", numerator.Grad().Data)
	fmt.Printf("Denominator Gradients: %v\n", denominator.Grad().Data)

	// Test parameter update
	fmt.Println("\nUpdating parameters...")
	lr := float32(0.1)
	denomData := denominator.value.Data
	denomGrad := denominator.Grad().Data
	for i := range denomData {
		denomData[i] -= lr * denomGrad[i]
	}

	fmt.Printf("Updated Denominator: %v\n", denomData)

	// Forward pass with updated denominator
	graph.Forward()
	fmt.Println("\nAfter Update Forward Pass:")
	fmt.Printf("New Quotient: %v\n", quotient.value.Data)
}
