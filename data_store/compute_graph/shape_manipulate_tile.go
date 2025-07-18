package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

func tileShape(shape, repeats []int) []int {
	if len(shape) != len(repeats) {
		panic("shape and repeats must have same length")
	}
	newShape := make([]int, len(shape))
	for i := range shape {
		newShape[i] = shape[i] * repeats[i]
	}
	return newShape
}

func tileTensor(a *tensor.Tensor, repeats []int) *tensor.Tensor {
	inputShape := a.GetShape()
	outputShape := tileShape(inputShape, repeats)
	outputData := make([]float32, prod(outputShape))

	strides := make([]int, len(inputShape))
	stride := 1
	for i := len(inputShape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= inputShape[i]
	}

	outputStrides := make([]int, len(outputShape))
	stride = 1
	for i := len(outputShape) - 1; i >= 0; i-- {
		outputStrides[i] = stride
		stride *= outputShape[i]
	}

	for i := range outputData {
		coord := make([]int, len(outputShape))
		temp := i
		for j := len(outputShape) - 1; j >= 0; j-- {
			coord[j] = temp / outputStrides[j]
			temp %= outputStrides[j]
		}

		inputIndex := 0
		for j := range coord {
			inputIndex += (coord[j] % inputShape[j]) * strides[j]
		}

		outputData[i] = a.Data[inputIndex]
	}

	return tensor.NewTensor(outputData, outputShape)
}

func untileTensor(grad *tensor.Tensor, inputShape []int) *tensor.Tensor {
	inputData := make([]float32, prod(inputShape))

	strides := make([]int, len(inputShape))
	stride := 1
	for i := len(inputShape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= inputShape[i]
	}

	gradStrides := make([]int, len(grad.GetShape()))
	stride = 1
	for i := len(grad.GetShape()) - 1; i >= 0; i-- {
		gradStrides[i] = stride
		stride *= grad.GetShape()[i]
	}

	for i := 0; i < len(grad.Data); i++ {
		coord := make([]int, len(grad.GetShape()))
		temp := i
		for j := len(grad.GetShape()) - 1; j >= 0; j-- {
			coord[j] = temp / gradStrides[j]
			temp %= gradStrides[j]
		}

		inputIndex := 0
		for j := range coord {
			inputIndex += (coord[j] % inputShape[j]) * strides[j]
		}

		inputData[inputIndex] += grad.Data[i]
	}

	return tensor.NewTensor(inputData, inputShape)
}

type Tile struct {
	OPS
	Repeats []int
}

func (m *Tile) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	result := tileTensor(a, m.Repeats)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Tile) Backward(grad *tensor.Tensor) {
	if grad == nil {
		panic("nil gradient in tile backward pass")
	}

	inputShape := m.Children[0].value.GetShape()
	gradTensor := untileTensor(grad, inputShape)
	m.Children[0].Node.Backward(gradTensor)
}

func (t *GraphTensor) Tile(repeats []int, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("tile_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	outputShape := tileShape(t.Shape, repeats)
	size := prod(outputShape)

	node := NewTile(name, t, repeats)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor(make([]float32, size), outputShape),
		grad:  tensor.NewTensor(make([]float32, size), outputShape),
		Shape: outputShape,
		Graph: g,
		Node:  node,
	}

	node.output = outputTensor

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewTile(name string, a *GraphTensor, repeats []int) *Tile {
	return &Tile{
		OPS: OPS{
			Name:     name,
			Children: []*GraphTensor{a},
		},
		Repeats: repeats,
	}
}
