package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ReduceMax struct {
	*OPSNode
	OPSTensor
}

func (m *ReduceMax) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()

	maxVal := input.Data[0]
	for _, val := range input.Data {
		if val > maxVal {
			maxVal = val
		}
	}

	result := tensor.NewTensor([]float32{maxVal}, []int{1})
	m.output.value = result
	m.output.computed = true
	return result
}

func NewReduceMax(name string, a *GraphTensor) *ReduceMax {
	return &ReduceMax{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "ReduceMax",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *ReduceMax) GetOutput() *tensor.Tensor {
	return m.output.value
}
