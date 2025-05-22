package tensor

func (t *Tensor) ZeroGrad() {
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}

func (t *Tensor) Backward() {
	if len(t.Grad) == 0 {
		t.Grad = make([]float32, len(t.Data))
	}
	if t.Grad[0] == 0 {
		t.Grad[0] = 1.0
	}

	visited := make(map[*Tensor]bool)
	var visit func(u *Tensor)
	visit = func(u *Tensor) {
		if visited[u] {
			return
		}
		visited[u] = true
		if u.GradFn != nil {
			u.GradFn()
		}
		for _, parent := range u.Parents {
			visit(parent)
		}
	}
	visit(t)
}

func (t *Tensor) RequireGrad() bool {
	return t.RequiresGrad
}
