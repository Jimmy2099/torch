package compute_graph

import (
	"fmt"
	"testing"
)

func TestCeil(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{1.2, 2.7, -1.8, 3.1}, []int{2, 2}, "input")
	ceil := input.Ceil("ceil_result")

	graph.SetOutput(ceil)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Ceil: %v\n", ceil.value.Data)

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
