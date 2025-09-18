package compute_graph

import (
	"fmt"
	"testing"
)

func TestConstant(t *testing.T) {
	graph := NewComputationalGraph()

	constantValue := []float32{1.0, 2.0, 3.0, 4.0}
	constantShape := []int{2, 2}
	output := graph.Constant(constantValue, constantShape, "constant_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Constant Output: %v\n", output.value.Data)
	fmt.Printf("Output Shape: %v\n", output.value.GetShape())

	for i, v := range output.value.Data {
		if v != constantValue[i] {
			t.Errorf("Constant value mismatch at index %d: expected %f, got %f", i, constantValue[i], v)
		}
	}

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Println("Backward pass completed successfully for constant node")
}
