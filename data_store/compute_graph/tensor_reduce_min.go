package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ReduceMin struct {
	*OPSNode
	OPSTensor
}

func (m *ReduceMin) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()

	minVal := input.Data[0]
	for _, val := range input.Data {
		if val < minVal {
			minVal = val
		}
	}

	result := tensor.NewTensor([]float32{minVal}, []int{1})
	m.output.value = result
	m.output.computed = true
	return result
}

func NewReduceMin(name string, a *GraphTensor) *ReduceMin {
	return &ReduceMin{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "ReduceMin",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *ReduceMin) GetOutput() *tensor.Tensor {
	return m.output.value
}
