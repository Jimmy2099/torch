package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Neg struct {
	*OPSNode
	OPSTensor
}

func (m *Neg) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}
	input := m.Children[0].Node.Forward()
	m.output.value = input.Negate()
	m.output.computed = true
	return m.output.value
}

func NewNeg(name string, a *GraphTensor) *Neg {
	return &Neg{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Neg",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Neg) GetOutput() *tensor.Tensor {
	return m.output.value
}
