package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Sqrt struct {
	*OPSNode
	OPSTensor
}

func (m *Sqrt) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}
	input := m.Children[0].Node.Forward()
	out := input.Clone().Sqrt()
	m.output.value = tensor.NewTensor(out.Data, input.GetShape())
	m.output.computed = true
	return m.output.value
}

func NewSqrt(name string, a *GraphTensor) *Sqrt {
	return &Sqrt{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Sqrt",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Sqrt) GetOutput() *tensor.Tensor {
	return m.output.value
}
