package compute_graph

import (
	"fmt"
	"testing"
)

func TestCos(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{0, 0.5, 1, 2}, []int{2, 2}, "input")
	output := input.Cos("cos_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Cos Output: %v\n", output.value.Data)

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
