package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ReLU struct {
	*OPSNode
	OPSTensor
}

func (m *ReLU) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	result := a.ReLU()
	m.output.value = result
	m.output.computed = true
	return result
}

func NewReLU(name string, a *GraphTensor) *ReLU {
	return &ReLU{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Relu",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *ReLU) GetOutput() *tensor.Tensor {
	return m.output.value
}
