package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Tanh struct {
	*OPSNode
	OPSTensor
}

func (m *Tanh) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	result := a.Tanh()
	m.output.value = result
	m.output.computed = true
	return result
}

func NewTanh(name string, a *GraphTensor) *Tanh {
	return &Tanh{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Tanh",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Tanh) GetOutput() *tensor.Tensor {
	return m.output.value
}
