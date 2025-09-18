package compute_graph

import (
	"fmt"
	"testing"
)

func TestReduceMean(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{3, 1, 4, 1, 5, 9}, []int{2, 3}, "input")
	output := input.ReduceMean("reduce_mean_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("ReduceMean Output: %v\n", output.value.Data)

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
