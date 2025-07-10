package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Ceil struct {
	OPS
}

func (m *Ceil) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].node.Forward()
	data := make([]float32, len(a.Data))
	for i, v := range a.Data {
		data[i] = float32(math.Ceil(float64(v)))
	}

	result := tensor.NewTensor(data, a.GetShape())
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Ceil) Backward(grad *tensor.Tensor) {
	gradInput := tensor.NewTensor(make([]float32, len(grad.Data)), grad.GetShape())
	m.Children[0].node.Backward(gradInput)
}

func (t *GraphTensor) Ceil(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("ceil_%d", t.graph.nodeCount)
		t.graph.nodeCount++
	}

	g := t.graph

	node := NewCeil(name, t)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		shape: t.shape,
		graph: g,
		node:  node,
	}

	if _, exists := g.tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.tensors[name] = outputTensor
	node.output = outputTensor
	g.nodes = append(g.nodes, node)
	return outputTensor
}

func NewCeil(name string, a *GraphTensor) *Ceil {
	return &Ceil{
		OPS{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}
