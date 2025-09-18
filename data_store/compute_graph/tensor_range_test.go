package compute_graph

import (
	"fmt"
	"testing"
)

func TestRange(t *testing.T) {
	graph := NewComputationalGraph()

	start := graph.NewGraphTensor([]float32{2}, []int{1}, "start")
	limit := graph.NewGraphTensor([]float32{10}, []int{1}, "limit")
	delta := graph.NewGraphTensor([]float32{2}, []int{1}, "delta")

	output := graph.Range("range_output", start, limit, delta)
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Range Output: %v\n", output.value.Data)

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Start Gradients: %v\n", start.Grad().Data)
	fmt.Printf("Limit Gradients: %v\n", limit.Grad().Data)
	fmt.Printf("Delta Gradients: %v\n", delta.Grad().Data)
}
