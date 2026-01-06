package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Flatten struct {
	*OPSNode
	OPSTensor
	originalShape []int
}

func (m *Flatten) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	m.originalShape = input.GetShape()
	totalSize := 1
	for _, dim := range m.originalShape {
		totalSize *= dim
	}

	result := input.Reshape([]int{totalSize})
	m.output.value = result
	m.output.computed = true
	return result
}

func NewFlatten(name string, a *GraphTensor) *Flatten {
	return &Flatten{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Flatten",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Flatten) GetOutput() *tensor.Tensor {
	return m.output.value
}
