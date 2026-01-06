package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ShapeOp struct {
	*OPSNode
	OPSTensor
}

func (m *ShapeOp) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	shape := a.GetShape()
	shapeData := make([]float32, len(shape))
	for i, dim := range shape {
		shapeData[i] = float32(dim)
	}
	result := tensor.NewTensor(shapeData, []int{len(shape)})
	m.output.value = result
	m.output.computed = true
	return result
}

func NewShapeOp(name string, a *GraphTensor) *ShapeOp {
	return &ShapeOp{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Shape",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *ShapeOp) GetOutput() *tensor.Tensor {
	return m.output.value
}
