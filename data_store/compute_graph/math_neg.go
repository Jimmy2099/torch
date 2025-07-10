package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Neg struct {
	OPS
}

func (m *Neg) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}
	input := m.Children[0].node.Forward()
	m.output.value = input.Negate()
	m.output.computed = true
	return m.output.value
}

func (m *Neg) Backward(grad *tensor.Tensor) {
	m.Children[0].node.Backward(grad.Negate())
}

func (t *GraphTensor) Neg(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("neg_%d", t.graph.nodeCount)
		t.graph.nodeCount++
	}

	g := t.graph

	node := NewNeg(name, t)

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

func NewNeg(name string, a *GraphTensor) *Neg {
	return &Neg{
		OPS{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}
