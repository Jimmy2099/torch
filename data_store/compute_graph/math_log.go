package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Log struct {
	*OPSNode
	OPSTensor
}

func (m *Log) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}
	input := m.Children[0].Node.Forward()
	data := make([]float32, len(input.Data))
	for i, v := range input.Data {
		if v <= 0 {
			panic("log input must be positive")
		}
		data[i] = float32(math.Log(float64(v)))
	}
	m.output.value = tensor.NewTensor(data, input.GetShape())
	m.output.computed = true
	return m.output.value
}

func (m *Log) GetOutput() *tensor.Tensor {
	return m.output.value
}
