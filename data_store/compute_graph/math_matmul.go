package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type MatMul struct {
	*OPSNode
	OPSTensor
}

func (m *MatMul) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	b := m.Children[1].Node.Forward()

	if len(a.GetShape()) < 2 || len(b.GetShape()) < 2 {
		panic("MatMul requires 2D tensors")
	}
	if a.GetShape()[len(a.GetShape())-1] != b.GetShape()[len(b.GetShape())-2] {
		panic(fmt.Sprintf("Incompatible dimensions for MatMul: %v and %v",
			a.GetShape(), b.GetShape()))
	}

	result := a.MatMul(b)
	m.output.value = result
	m.output.computed = true
	return result
}

func NewMatMul(name string, a, b *GraphTensor) *MatMul {
	return &MatMul{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "MatMul",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a, b},
		},
	}
}

func (m *MatMul) GetOutput() *tensor.Tensor {
	return m.output.value
}
