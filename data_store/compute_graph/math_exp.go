package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Exp struct {
	OPS
}

func (m *Exp) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].node.Forward()
	result := a.Exp()
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Exp) Backward(grad *tensor.Tensor) {
	aVal := m.Children[0].value
	if aVal == nil || grad == nil {
		panic("nil tensor in exponential backward pass")
	}

	// dc/da = exp(a) * grad
	gradA := m.output.value.Copy().Mul(grad)
	m.Children[0].node.Backward(gradA)
}

func (t *GraphTensor) Exp(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("div_%d", t.graph.nodeCount)
		t.graph.nodeCount++
	}

	g := t.graph

	node := NewExp(name, t)

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

func NewExp(name string, a *GraphTensor) *Exp {
	return &Exp{
		OPS{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}
