package compute_graph

import (
	"fmt"
	"testing"
)

func TestScatterND(t *testing.T) {
	graph := NewComputationalGraph()

	data := graph.NewGraphTensor([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, "data")
	indices := graph.NewGraphTensor([]float32{1, 0, 2}, []int{3, 1}, "indices")
	updates := graph.NewGraphTensor([]float32{10, 20, 30}, []int{3}, "updates")

	output := data.ScatterND(indices, updates, "scatternd_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("ScatterND Output: %v\n", output.value.Data)

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Data Gradients: %v\n", data.Grad().Data)
	fmt.Printf("Indices Gradients: %v\n", indices.Grad().Data)
	fmt.Printf("Updates Gradients: %v\n", updates.Grad().Data)
}
