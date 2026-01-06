package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Cos struct {
	*OPSNode
	OPSTensor
}

func (m *Cos) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	resultData := make([]float32, len(a.Data))
	for i, val := range a.Data {
		resultData[i] = float32(math.Cos(float64(val)))
	}
	result := tensor.NewTensor(resultData, a.GetShape())
	m.output.value = result
	m.output.computed = true
	return result
}

func NewCos(name string, a *GraphTensor) *Cos {
	return &Cos{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Cos",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Cos) GetOutput() *tensor.Tensor {
	return m.output.value
}
