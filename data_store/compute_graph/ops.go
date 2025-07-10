package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type OPS struct {
	Name     string
	Children []*GraphTensor
	output   *GraphTensor
}

func (m *OPS) GetName() string { return m.Name }

func (m *OPS) GetChildren() []Node {
	nodes := make([]Node, len(m.Children))
	for i, t := range m.Children {
		nodes[i] = t.Node
	}
	return nodes
}

func (m *OPS) ResetComputed() {
	m.output.computed = false
}

func (m *OPS) GetOutput() *tensor.Tensor { return m.output.value }
