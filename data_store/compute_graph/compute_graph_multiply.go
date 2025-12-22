package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/node"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Multiply struct {
	*OPSNode
	Name     string
	Children []*GraphTensor
	output   *GraphTensor
}

func NewMultiply(name string, a, b *GraphTensor) *Multiply {
	return &Multiply{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Mul",
			ONNXProducedTensor: true,
		}),
		Name:     name,
		Children: []*GraphTensor{a, b},
	}
}

func (m *Multiply) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	b := m.Children[1].Node.Forward()

	result := a.Mul(b)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Multiply) ResetComputed() {
	m.output.computed = false
}

func (m *Multiply) GetName() string { return m.Name }

func (m *Multiply) GetChildren() []node.Node {
	nodes := make([]node.Node, len(m.Children))
	for i, t := range m.Children {
		nodes[i] = t.Node
	}
	return nodes
}

func (m *Multiply) GetOutput() *tensor.Tensor { return m.output.value }
