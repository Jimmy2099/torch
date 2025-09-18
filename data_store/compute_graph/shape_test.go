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

func TestSlice(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensor
	input := graph.NewGraphTensor([]float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, []int{2, 3}, "input")

	// Create slice operation
	sliced := input.Slice([]int{0, 1}, []int{2, 3}, "slice_result")

	graph.SetOutput(sliced)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Sliced Data: %v\n", sliced.value.Data)
	fmt.Printf("Sliced Shape: %v\n", sliced.value.GetShape())

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)

	// Test parameter update
	fmt.Println("\nUpdating parameters...")
	lr := float32(0.1)
	inputData := input.value.Data
	inputGrad := input.Grad().Data
	for i := range inputData {
		inputData[i] -= lr * inputGrad[i]
	}

	fmt.Printf("Updated Input: %v\n", inputData)
}

func TestTile(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensor
	input := graph.NewGraphTensor([]float32{1.0, 2.0}, []int{2}, "input")

	// Create tile operation
	tiled := input.Tile([]int{3}, "tile_result")

	graph.SetOutput(tiled)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Tiled Data: %v\n", tiled.value.Data)
	fmt.Printf("Tiled Shape: %v\n", tiled.value.GetShape())

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)

	// Test parameter update
	fmt.Println("\nUpdating parameters...")
	lr := float32(0.1)
	inputData := input.value.Data
	inputGrad := input.Grad().Data
	for i := range inputData {
		inputData[i] -= lr * inputGrad[i]
	}

	fmt.Printf("Updated Input: %v\n", inputData)
}

func TestSqueeze(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensor with shape [1, 3, 1, 2]
	input := graph.NewGraphTensor([]float32{1, 2, 3, 4, 5, 6}, []int{1, 3, 1, 2}, "input")
	squeezed := input.Squeeze("squeezed")

	graph.SetOutput(squeezed)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Squeezed shape: %v\n", squeezed.value.GetShape())
	fmt.Printf("Squeezed data: %v\n", squeezed.value.Data)

	// Backward pass
	squeezed.Grad().Data = []float32{1, 1, 1, 1, 1, 1} // dL/dsqueezed
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input gradients: %v\n", input.Grad().Data)
}

func TestUnsqueeze(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensor with shape [2, 3]
	input := graph.NewGraphTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, "input")
	unsqueezed := input.Unsqueeze(1, "unsqueezed") // Add dimension at axis 1

	graph.SetOutput(unsqueezed)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Unsqueezed shape: %v\n", unsqueezed.value.GetShape())
	fmt.Printf("Unsqueezed data: %v\n", unsqueezed.value.Data)

	// Backward pass
	unsqueezed.Grad().Data = []float32{1, 1, 1, 1, 1, 1} // dL/dunsqueezed
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input gradients: %v\n", input.Grad().Data)
}

func TestPad(t *testing.T) {
	graph := NewComputationalGraph()

	// Create input tensor
	input := graph.NewGraphTensor([]float32{1.0, 2.0}, []int{2}, "input")

	// Create pad operation
	padded := input.Pad([][2]int{{1, 2}}, 0.0, "pad_result")

	graph.SetOutput(padded)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	// Forward pass
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Padded Data: %v\n", padded.value.Data)
	fmt.Printf("Padded Shape: %v\n", padded.value.GetShape())

	// Backward pass
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)

	// Test parameter update
	fmt.Println("\nUpdating parameters...")
	lr := float32(0.1)
	inputData := input.value.Data
	inputGrad := input.Grad().Data
	for i := range inputData {
		inputData[i] -= lr * inputGrad[i]
	}

	fmt.Printf("Updated Input: %v\n", inputData)
}
