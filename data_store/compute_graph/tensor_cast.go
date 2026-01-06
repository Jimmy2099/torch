package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Cast struct {
	*OPSNode
	OPSTensor
}

func (m *Cast) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	m.output.value = a
	m.output.computed = true
	return a
}

func NewCast(name string, a *GraphTensor) *Cast {
	return &Cast{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Cast",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Cast) GetOutput() *tensor.Tensor {
	return m.output.value
}
