package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Reciprocal struct {
	*OPSNode
	OPSTensor
}

func (m *Reciprocal) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()

	ones := tensor.NewTensor(make([]float32, len(a.Data)), a.GetShape())
	for i := range ones.Data {
		ones.Data[i] = 1.0
	}

	result := ones.Div(a)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Reciprocal) GetOutput() *tensor.Tensor {
	return m.output.value
}
