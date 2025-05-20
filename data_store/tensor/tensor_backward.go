package tensor

func (t *Tensor) ZeroGrad() {
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}
