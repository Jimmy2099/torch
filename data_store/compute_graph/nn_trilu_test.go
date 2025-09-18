package compute_graph

import (
	"fmt"
	"testing"
)

func TestTrilu(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{3, 3}, "input")
	k := graph.NewGraphTensor([]float32{0}, []int{1}, "k")
	output := input.Trilu(k, "trilu_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Trilu Output: %v\n", output.value.Data)

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
	fmt.Printf("K Gradients: %v\n", k.Grad().Data)
}
