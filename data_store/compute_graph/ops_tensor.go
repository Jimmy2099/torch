package compute_graph

import "github.com/Jimmy2099/torch/data_store/node"

type OPSTensor struct {
	Name     string
	Children []*GraphTensor
	output   *GraphTensor
}

func (m *OPSTensor) GetName() string { return m.Name }

func (m *OPSTensor) GetChildren() []node.Node {
	nodes := make([]node.Node, len(m.Children))
	for i, t := range m.Children {
		nodes[i] = t.Node
	}
	return nodes
}

func (m *OPSTensor) ResetComputed() {
	m.output.computed = false
}

func (m *OPSTensor) GetTensorOutput() *GraphTensor { return m.output }
