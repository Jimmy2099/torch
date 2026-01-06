package compute_graph

import (
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

func (m *Exp) GetOutput() *tensor.Tensor {
	return m.output.value
}
