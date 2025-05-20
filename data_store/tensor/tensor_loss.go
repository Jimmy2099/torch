package tensor

func (t *Tensor) LossMSE(other *Tensor) float32 {
	if other == nil {
		panic("other tensor is nil")
	}
	if t.Device != other.Device {
		panic("tensors must be on the same device")
	}
	if len(t.shape) != len(other.shape) {
		panic("tensors must have the same shape")
	}
	for i := range t.shape {
		if t.shape[i] != other.shape[i] {
			panic("tensors must have the same shape")
		}
	}
	if len(t.Data) != len(other.Data) {
		panic("data length mismatch")
	}
	if len(t.Data) == 0 {
		panic("tensor is empty")
	}

	var sum float32
	for i := range t.Data {
		diff := t.Data[i] - other.Data[i]
		sum += diff * diff
	}
	return sum / float32(len(t.Data))
}
