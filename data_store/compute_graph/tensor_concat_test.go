package compute_graph

import (
	"fmt"
	"testing"
)

func TestConcat(t *testing.T) {
	graph := NewComputationalGraph()

	input1 := graph.NewGraphTensor([]float32{1, 2, 3}, []int{3}, "input1")
	input2 := graph.NewGraphTensor([]float32{4, 5, 6}, []int{3}, "input2")
	input3 := graph.NewGraphTensor([]float32{7, 8, 9}, []int{3}, "input3")

	output := input1.Concat([]*GraphTensor{input2, input3}, 0, "concat_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Concat Output: %v\n", output.value.Data)
	fmt.Printf("Output Shape: %v\n", output.value.GetShape())

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input1 Gradients: %v\n", input1.Grad().Data)
	fmt.Printf("Input2 Gradients: %v\n", input2.Grad().Data)
	fmt.Printf("Input3 Gradients: %v\n", input3.Grad().Data)
}

func TestConcatDifferentAxis(t *testing.T) {
	graph := NewComputationalGraph()

	input1 := graph.NewGraphTensor([]float32{1, 2, 3, 4}, []int{2, 2}, "input1")
	input2 := graph.NewGraphTensor([]float32{5, 6, 7, 8}, []int{2, 2}, "input2")

	outputAxis0 := input1.Concat([]*GraphTensor{input2}, 0, "concat_axis0")
	graph.SetOutput(outputAxis0)

	graph.Forward()
	fmt.Println("\nConcat along axis 0:")
	fmt.Printf("Output: %v\n", outputAxis0.value.Data)
	fmt.Printf("Output Shape: %v\n", outputAxis0.value.GetShape())

	outputAxis1 := input1.Concat([]*GraphTensor{input2}, 1, "concat_axis1")
	graph.SetOutput(outputAxis1)

	graph.Forward()
	fmt.Println("\nConcat along axis 1:")
	fmt.Printf("Output: %v\n", outputAxis1.value.Data)
	fmt.Printf("Output Shape: %v\n", outputAxis1.value.GetShape())
}
