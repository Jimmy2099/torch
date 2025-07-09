package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Flatten struct {
	OPS
	originalShape []int
}

func (m *Flatten) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	m.originalShape = input.GetShape()
	totalSize := 1
	for _, dim := range m.originalShape {
		totalSize *= dim
	}

	result := input.Reshape([]int{totalSize})
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Flatten) Backward(grad *tensor.Tensor) {
	reshapedGrad := grad.Reshape(m.originalShape)
	m.Children[0].Node.Backward(reshapedGrad)
}

func (t *GraphTensor) Flatten(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("flatten_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewFlatten(name, t)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Shape: []int{0},
		Graph: g,
		Node:  node,
	}

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewFlatten(name string, a *GraphTensor) *Flatten {
	return &Flatten{
		OPS: OPS{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}
