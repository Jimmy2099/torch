package compute_graph

import (
	"fmt"
	"testing"
)

func TestLpNormalization(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor(
		[]float32{1, 2, 3, 4},
		[]int{2, 2},
		"input",
	)

	output := input.LpNormalization(2.0, 1, 1e-5, "lp_norm")

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
