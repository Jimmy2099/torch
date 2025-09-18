package compute_graph

import (
	"fmt"
	"testing"
)

func TestConstantOfShape(t *testing.T) {
	graph := NewComputationalGraph()

	shapeTensor := graph.NewGraphTensor([]float32{2, 3}, []int{2}, "shape_input")

	output := shapeTensor.ConstantOfShape("constant_of_shape_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("ConstantOfShape Output: %v\n", output.value.Data)
	fmt.Printf("Output Shape: %v\n", output.value.GetShape())

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Shape Input Gradients: %v\n", shapeTensor.Grad().Data)
}
