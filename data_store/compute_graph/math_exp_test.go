package compute_graph

import (
	"fmt"
	"math"
	"testing"
)

func TestExp(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensor
	input := graph.NewGraphTensor([]float32{1.0, 2.0, 0.0, -1.0}, []int{2, 2}, "input")

	// Exponential operation
	expResult := input.Exp("exp_result")
	graph.SetOutput(expResult)

	fmt.Println("Exponential Computation Graph:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass (Exp):")
	fmt.Printf("Input: %v\n", input.Value().Data)
	fmt.Printf("Exp Result: %v\n", expResult.Value().Data)

	// Validate results
	expected := []float32{
		float32(math.Exp(1.0)),
		float32(math.Exp(2.0)),
		float32(math.Exp(0.0)),
		float32(math.Exp(-1.0)),
	}
	validateResults(t, "Exp forward", expResult.Value().Data, expected)

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass (Exp):")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)

	// Validate gradients
	expectedGrad := []float32{
		float32(math.Exp(1.0)),
		float32(math.Exp(2.0)),
		float32(math.Exp(0.0)),
		float32(math.Exp(-1.0)),
	}
	validateResults(t, "Exp backward", input.Grad().Data, expectedGrad)
}
