package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Unsqueeze struct {
	*OPSNode
	OPSTensor
	originalShape []int
	axis          int
}

func (m *Unsqueeze) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	result := input.UnSqueeze(m.axis)
	return result
}

func NewUnsqueeze(name string, children []*GraphTensor, axis int) *Unsqueeze {

	return &Unsqueeze{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Unsqueeze",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: children,
		},
		axis: axis,
	}
}

func (m *Unsqueeze) GetOutput() *tensor.Tensor {
	return m.output.value
}
