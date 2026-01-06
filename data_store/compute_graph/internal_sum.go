package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/node"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Sum struct {
	Name     string
	Children []*GraphTensor
	output   *GraphTensor
}

func (m *Sum) GetONNXNodeInfo() *node.ONNXNodeInfo {
	return &node.ONNXNodeInfo{
		Name:           "Sum",
		ProducedTensor: true,
	}
}

func NewSum(name string, a *GraphTensor) *Sum {
	return &Sum{
		Name:     name,
		Children: []*GraphTensor{a},
	}
}

func (m *Sum) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()

	sum := float32(0)
	for _, val := range a.Data {
		sum += val
	}

	result := tensor.NewTensor([]float32{sum}, []int{1})
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Sum) ResetComputed() {
	m.output.computed = false
}

func (m *Sum) GetName() string { return m.Name }

func (m *Sum) GetChildren() []node.Node {
	nodes := make([]node.Node, len(m.Children))
	for i, t := range m.Children {
		nodes[i] = t.Node
	}
	return nodes
}

func (m *Sum) GetOutput() *tensor.Tensor {
	return m.output.value
}
