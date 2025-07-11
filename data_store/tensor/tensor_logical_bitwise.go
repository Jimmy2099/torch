package tensor

// And performs element-wise logical AND operation
func (t *Tensor) And(other *Tensor) *Tensor {
	if !t.ShapesMatch(other) {
		panic("tensor shapes must match for AND operation")
	}

	data := make([]float32, len(t.Data))
	for i := range t.Data {
		// Treat non-zero as true, zero as false
		a := 0.0
		if t.Data[i] != 0 {
			a = 1.0
		}

		b := 0.0
		if other.Data[i] != 0 {
			b = 1.0
		}

		// AND operation: 1 if both are non-zero, else 0
		if a != 0 && b != 0 {
			data[i] = 1.0
		} else {
			data[i] = 0.0
		}
	}
	return NewTensor(data, t.GetShape())
}

// Or performs element-wise logical OR operation
func (t *Tensor) Or(other *Tensor) *Tensor {
	if !t.ShapesMatch(other) {
		panic("tensor shapes must match for OR operation")
	}

	data := make([]float32, len(t.Data))
	for i := range t.Data {
		// Treat non-zero as true, zero as false
		a := 0.0
		if t.Data[i] != 0 {
			a = 1.0
		}

		b := 0.0
		if other.Data[i] != 0 {
			b = 1.0
		}

		// OR operation: 1 if either is non-zero, else 0
		if a != 0 || b != 0 {
			data[i] = 1.0
		} else {
			data[i] = 0.0
		}
	}
	return NewTensor(data, t.GetShape())
}

// Xor performs element-wise logical XOR operation
func (t *Tensor) Xor(other *Tensor) *Tensor {
	if !t.ShapesMatch(other) {
		panic("tensor shapes must match for XOR operation")
	}

	data := make([]float32, len(t.Data))
	for i := range t.Data {
		// Treat non-zero as true, zero as false
		a := 0.0
		if t.Data[i] != 0 {
			a = 1.0
		}

		b := 0.0
		if other.Data[i] != 0 {
			b = 1.0
		}

		// XOR operation: 1 if values are different, else 0
		if (a != 0) != (b != 0) {
			data[i] = 1.0
		} else {
			data[i] = 0.0
		}
	}
	return NewTensor(data, t.GetShape())
}

// Not performs element-wise logical NOT operation
func (t *Tensor) Not() *Tensor {
	data := make([]float32, len(t.Data))
	for i := range t.Data {
		// Treat non-zero as true, zero as false
		// NOT operation: 1 if zero, 0 if non-zero
		if t.Data[i] == 0 {
			data[i] = 1.0
		} else {
			data[i] = 0.0
		}
	}
	return NewTensor(data, t.GetShape())
}
