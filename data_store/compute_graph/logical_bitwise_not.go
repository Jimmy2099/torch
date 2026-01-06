package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Not struct {
	*OPSNode
	OPSTensor
}

func (m *Not) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()

	result := a.Not()
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Not) GetOutput() *tensor.Tensor {
	return m.output.value
}
