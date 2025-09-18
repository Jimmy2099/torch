package compute_graph

import (
	"fmt"
	"testing"
)

func TestExpand(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{1, 2}, []int{1, 2}, "input")

	shape := graph.NewGraphTensor([]float32{2, 2}, []int{2}, "shape")

	output := input.Expand(shape, "expand_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Input: %v\n", input.value.Data)
	fmt.Printf("Shape: %v\n", shape.value.Data)
	fmt.Printf("Expand Output: %v\n", output.value.Data)
	fmt.Printf("Output Shape: %v\n", output.value.GetShape())

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
	fmt.Printf("Shape Gradients: %v\n", shape.Grad().Data)
}
