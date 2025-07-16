package compute_graph

import (
	"fmt"
	"testing"
)

func TestLpPool(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor(
		[]float32{1, 2, 3, 4, 5, 6, 7, 8},
		[]int{1, 2, 2, 2}, // NCHW format
		"input",
	)

	output := input.LpPool([]int{2, 2}, []int{2, 2}, 2.0, "lp_pool")

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
