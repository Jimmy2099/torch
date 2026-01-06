package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Squeeze struct {
	*OPSNode
	OPSTensor
	originalShape []int
}

func (s *Squeeze) Forward() *tensor.Tensor {
	if s.output.computed {
		return s.output.value
	}

	input := s.Children[0].Node.Forward()
	s.originalShape = input.GetShape()

	newShape := []int{}
	for _, dim := range s.originalShape {
		if dim != 1 {
			newShape = append(newShape, dim)
		}
	}

	result := input.Reshape(newShape)
	s.output.value = result
	s.output.computed = true
	return result
}

func NewSqueeze(name string, a *GraphTensor) *Squeeze {
	return &Squeeze{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Squeeze",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Squeeze) GetOutput() *tensor.Tensor {
	return m.output.value
}
