package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ConstantOfShape struct {
	*OPSNode
	OPSTensor
	value float32
}

func (m *ConstantOfShape) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	shapeTensor := m.Children[0].Node.Forward()

	shape := make([]int, len(shapeTensor.Data))
	for i, val := range shapeTensor.Data {
		shape[i] = int(val)
	}

	numElements := 1
	for _, dim := range shape {
		numElements *= dim
	}

	data := make([]float32, numElements)
	for i := range data {
		data[i] = m.value
	}

	result := tensor.NewTensor(data, shape)
	m.output.value = result
	m.output.computed = true
	return result
}

func NewConstantOfShape(name string, shapeTensor *GraphTensor, value float32) *ConstantOfShape {
	return &ConstantOfShape{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "ConstantOfShape",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{shapeTensor},
		},
		value: value,
	}
}

func (m *ConstantOfShape) GetOutput() *tensor.Tensor {
	return m.output.value
}
