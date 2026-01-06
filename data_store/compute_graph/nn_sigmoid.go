package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Sigmoid struct {
	*OPSNode
	OPSTensor
}

func (m *Sigmoid) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	result := a.Sigmoid()
	m.output.value = result
	m.output.computed = true
	return result
}

func NewSigmoid(name string, a *GraphTensor) *Sigmoid {
	return &Sigmoid{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Sigmoid",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Sigmoid) GetOutput() *tensor.Tensor {
	return m.output.value
}
