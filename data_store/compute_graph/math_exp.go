package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Exp struct {
	*OPSNode
	OPSTensor
}

func (m *Exp) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
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
	m.Children[0].Node.Backward(gradA)
}

func (t *GraphTensor) Exp(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("div_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph

	node := NewExp(name, t)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Shape: t.Shape,
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

func NewExp(name string, a *GraphTensor) *Exp {
	return &Exp{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Exp",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{Name: name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Exp) GetOutput() *GraphTensor {
	return m.output
}
