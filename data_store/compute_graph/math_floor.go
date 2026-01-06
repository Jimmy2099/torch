package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Floor struct {
	*OPSNode
	OPSTensor
}

func (m *Floor) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	data := make([]float32, len(a.Data))
	for i, v := range a.Data {
		data[i] = float32(math.Floor(float64(v)))
	}

	result := tensor.NewTensor(data, a.GetShape())
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Floor) GetOutput() *tensor.Tensor {
	return m.output.value
}
