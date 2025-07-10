package compute_graph

import (
	"fmt"
	"testing"
)

func TestReciprocal(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{2.0, 4.0, 0.5, 0.25}, []int{2, 2}, "input")
	reciprocal := input.Reciprocal("reciprocal_result")

	graph.SetOutput(reciprocal)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Reciprocal: %v\n", reciprocal.value.Data)

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
