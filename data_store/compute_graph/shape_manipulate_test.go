package compute_graph

import (
	"fmt"
	"testing"
)

func TestFlatten(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, "input")

	flat := input.Flatten("flattened")

	graph.SetOutput(flat)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Reshaped: %v Shape:%v \n", flat.value.Data, flat.value.GetShape())

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v Shape:%v \n", input.Grad().Data, input.value.GetShape())
}

func TestReshape(t *testing.T) {
	graph := NewComputationalGraph()

	input := graph.NewGraphTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, "input")

	reshaped := input.Reshape([]int{3, 2}, "reshaped")

	graph.SetOutput(reshaped)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Reshaped: %v Shape:%v \n", reshaped.value.Data, reshaped.value.GetShape())

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v Shape:%v \n", input.Grad().Data, input.value.GetShape())
}

func TestTranspose(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensor
	input := graph.NewGraphTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, "input")

	// Transpose operation
	transposed := input.Transpose([]int{1, 0}, "transposed")

	graph.SetOutput(transposed)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Transposed: %v\n", transposed.value.Data)

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
}
