package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"testing"
)

func TestSoftmax(t *testing.T) {
	graph := NewComputationalGraph()
	input := graph.NewGraphTensor([]float32{1.0, 2.0, 3.0, 4.0}, []int{2, 2}, "input")
	output := input.Softmax("softmax_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Softmax Output: %v\n", output.value.Data)

	graph.Backward()
	fmt.Println("\nAfter Backward Pass (default grad):")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)

	// Custom gradient test
	graph.Reset()
	output.grad = tensor.NewTensor([]float32{1, 0, 0, 0}, []int{2, 2})
	graph.Backward()
	fmt.Println("\nAfter Backward Pass (custom grad [1,0,0,0]):")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
