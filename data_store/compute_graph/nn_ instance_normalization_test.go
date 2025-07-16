package compute_graph

import (
	"fmt"
	"testing"
)

func TestInstanceNormalization(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor(
		[]float32{1, 2, 3, 4, 5, 6, 7, 8},
		[]int{2, 2, 1, 2}, // NCHW format
		"input",
	)

	output := input.InstanceNormalization(1e-5, "inst_norm")

	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Output: %v\n", output.value.Data)

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
