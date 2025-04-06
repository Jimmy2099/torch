package tensor

type Tensor struct {
	Data  []float32
	Shape []int // e.g., [batch_size, channels, height, width]
}

func NewTensor(data []float32, shape []int) *Tensor {
	t := &Tensor{Data: data, Shape: shape}
	if shape == nil {
		shape = []int{1, len(data)}
		t.Reshape(shape)
	}
	return t
}

func (m *Tensor) TensorData() []float32 {
	return m.Data
}
