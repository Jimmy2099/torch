package tensor

func (pred *Tensor) LossMSE(target *Tensor) *Tensor {
	if len(pred.Data) != len(target.Data) {
		panic("MSE: shape mismatch")
	}
	var sum float32
	N := float32(len(pred.Data))
	for i := range pred.Data {
		d := pred.Data[i] - target.Data[i]
		sum += d * d
	}
	lossVal := sum / N
	out := NewTensor([]float32{lossVal}, []int{1, 1})
	out.EnableGrad()

	out.Parents = []*Tensor{pred}

	out.GradFn = func() {
		if !pred.RequiresGrad {
			return
		}
		for i := range pred.Data {
			d := pred.Data[i] - target.Data[i]
			grad := (2 * d / N) * out.Grad[0]
			pred.Grad[i] += grad
		}
	}
	return out
}
