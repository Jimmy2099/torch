package tensor

type Tensor struct {
	Data  []float64
	Shape []int // e.g., [batch_size, channels, height, width]
}

func NewTensor(data []float64, shape []int) *Tensor {
	return &Tensor{Data: data, Shape: shape}
}

func (m *Tensor) TensorData() []float64 {
	return m.Data
}
