package compute_graph

import (
	"fmt"
	"math"
	"testing"
)

func TestPow(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensors
	base := graph.NewGraphTensor([]float32{2.0, 3.0, 4.0, 1.0}, []int{2, 2}, "base")
	exponent := graph.NewGraphTensor([]float32{3.0, 2.0, 0.5, 10.0}, []int{2, 2}, "exponent")

	// Tensor Pow operation
	powerResult := base.Pow(exponent, "power_result")
	graph.SetOutput(powerResult)

	fmt.Println("\nPow Computation Graph:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass (Pow):")
	fmt.Printf("Base: %v\n", base.value.Data)
	fmt.Printf("Exponent: %v\n", exponent.value.Data)
	fmt.Printf("Pow Result: %v\n", powerResult.value.Data)

	// Validate results
	expected := []float32{
		float32(math.Pow(2.0, 3.0)),
		float32(math.Pow(3.0, 2.0)),
		float32(math.Pow(4.0, 0.5)),
		float32(math.Pow(1.0, 10.0)),
	}
	validateResults(t, "Pow forward", powerResult.value.Data, expected)

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass (Pow):")
	fmt.Printf("Base Gradients: %v\n", base.Grad().Data)
	fmt.Printf("Exponent Gradients: %v\n", exponent.Grad().Data)

	// Validate gradients
	expectedBaseGrad := []float32{
		3.0 * float32(math.Pow(2.0, 2.0)),  // 3 * 2^(3-1) = 3*4=12
		2.0 * float32(math.Pow(3.0, 1.0)),  // 2 * 3^(2-1) = 2*3=6
		0.5 * float32(math.Pow(4.0, -0.5)), // 0.5 * 4^(-0.5) = 0.5 * 0.5 = 0.25
		10.0 * float32(math.Pow(1.0, 9.0)), // 10 * 1^9 = 10
	}
	expectedExpGrad := []float32{
		float32(math.Pow(2.0, 3.0) * math.Log(2.0)),  // 8 * ln(2)
		float32(math.Pow(3.0, 2.0) * math.Log(3.0)),  // 9 * ln(3)
		float32(math.Pow(4.0, 0.5) * math.Log(4.0)),  // 2 * ln(4)
		float32(math.Pow(1.0, 10.0) * math.Log(1.0)), // 1 * 0 = 0
	}
	validateResults(t, "Pow base grad", base.Grad().Data, expectedBaseGrad)
	validateResults(t, "Pow exponent grad", exponent.Grad().Data, expectedExpGrad)
}

func validateResults(t *testing.T, testName string, actual, expected []float32) {
	tolerance := float32(1e-5)
	for i := range actual {
		if math.Abs(float64(actual[i]-expected[i])) > float64(tolerance) {
			t.Errorf("%s failed at index %d: got %v, want %v",
				testName, i, actual[i], expected[i])
		}
	}
}
