package compute_graph

import (
	"fmt"
	"testing"
)

func TestFloor(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{1.7, 2.3, -1.2, 3.8}, []int{2, 2}, "input")
	floor := input.Floor("floor_result")

	graph.SetOutput(floor)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Floor: %v\n", floor.value.Data)

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
