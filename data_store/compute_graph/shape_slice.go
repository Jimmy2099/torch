package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

func prod(shape []int) int {
	p := 1
	for _, dim := range shape {
		p *= dim
	}
	return p
}

func sliceShape(shape, starts, ends []int) []int {
	newShape := make([]int, len(shape))
	for i := range shape {
		newShape[i] = ends[i] - starts[i]
	}
	return newShape
}

type Slice struct {
	*OPSNode
	OPSTensor
	Starts []int
	Ends   []int
}

func (m *Slice) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	result := a
	for dim := 0; dim < len(m.Starts); dim++ {
		result = result.Slice(m.Starts[dim], m.Ends[dim], dim)
	}
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Slice) Backward(grad *tensor.Tensor) {
	if grad == nil {
		panic("nil gradient in slice backward pass")
	}

	inputShape := m.Children[0].value.GetShape()
	zeroData := make([]float32, prod(inputShape))
	zeroGrad := tensor.NewTensor(zeroData, inputShape)
	gradData := grad.Data

	ndim := len(inputShape)
	sliceShape := sliceShape(inputShape, m.Starts, m.Ends)

	strides := make([]int, ndim)
	stride := 1
	for i := ndim - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= inputShape[i]
	}

	for idx := 0; idx < len(gradData); idx++ {
		coord := make([]int, ndim)
		remainder := idx
		for i := ndim - 1; i >= 0; i-- {
			coord[i] = remainder % sliceShape[i]
			remainder /= sliceShape[i]
		}

		inputCoord := make([]int, ndim)
		for i := range coord {
			inputCoord[i] = coord[i] + m.Starts[i]
		}

		inputIndex := 0
		for i := range inputCoord {
			inputIndex += inputCoord[i] * strides[i]
		}

		zeroData[inputIndex] = gradData[idx]
	}

	m.Children[0].Node.Backward(zeroGrad)
}

func (t *GraphTensor) Slice(starts, ends []int, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("slice_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	outputShape := sliceShape(t.Shape, starts, ends)
	size := prod(outputShape)

	node := NewSlice(name, t, starts, ends)

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

func NewSlice(name string, a *GraphTensor, starts, ends []int) *Slice {
	return &Slice{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Slice",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
		Starts: starts,
		Ends:   ends,
	}
}

func (m *Slice) GetOutput() *GraphTensor {
	return m.output
}
