package compute_graph

import (
	"fmt"
	"testing"
)

func TestShapeOp(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, "input")
	output := input.ShapeOp("shape_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Shape Output: %v\n", output.value.Data)

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
