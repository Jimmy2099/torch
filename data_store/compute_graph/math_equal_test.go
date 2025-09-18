package compute_graph

import (
	"fmt"
	"testing"
)

func TestEqual(t *testing.T) {
	graph := NewComputationalGraph()

	input1 := graph.NewGraphTensor([]float32{1, 2, 3, 4}, []int{2, 2}, "input1")
	input2 := graph.NewGraphTensor([]float32{1, 5, 3, 4}, []int{2, 2}, "input2")
	output := input1.Equal(input2, "equal_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Equal Output: %v\n", output.value.Data)

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input1 Gradients: %v\n", input1.Grad().Data)
	fmt.Printf("Input2 Gradients: %v\n", input2.Grad().Data)
}
