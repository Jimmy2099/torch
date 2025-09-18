package compute_graph

import (
	"fmt"
	"testing"
)

func TestGather(t *testing.T) {
	graph := NewComputationalGraph()

	data := graph.NewGraphTensor([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, "data")

	indices := graph.NewGraphTensor([]float32{0, 2}, []int{2}, "indices")

	output := data.Gather(indices, "gather_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Gather Output: %v\n", output.value.Data)
	fmt.Printf("Gather Output Shape: %v\n", output.value.GetShape())

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Data Gradients: %v\n", data.Grad().Data)
}
