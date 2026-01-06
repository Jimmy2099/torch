package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ReduceMean struct {
	*OPSNode
	OPSTensor
}

func (m *ReduceMean) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()

	sum := float32(0)
	for _, val := range input.Data {
		sum += val
	}
	mean := sum / float32(len(input.Data))

	result := tensor.NewTensor([]float32{mean}, []int{1})
	m.output.value = result
	m.output.computed = true
	return result
}

func NewReduceMean(name string, a *GraphTensor) *ReduceMean {
	return &ReduceMean{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "ReduceMean",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *ReduceMean) GetOutput() *tensor.Tensor {
	return m.output.value
}
