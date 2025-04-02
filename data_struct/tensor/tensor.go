package tensor

type Tensor struct {
	Data  []float64
	Shape []int // e.g., [batch_size, channels, height, width]
}

func NewTensor(data []float64, shape []int) *Tensor {
	t := &Tensor{Data: data, Shape: shape}
	if shape == nil {
		shape = []int{1, len(data)}
		t.Reshape(shape)
	}
	return t
}

func (m *Tensor) TensorData() []float64 {
	return m.Data
}
