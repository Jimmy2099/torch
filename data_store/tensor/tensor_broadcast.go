package tensor

func (t *Tensor) broadcastedIndex(indices []int, strides []int) int {
	index := 0
	for i := 0; i < len(indices); i++ {
		pos := len(t.shape) - len(indices) + i
		if pos >= 0 && pos < len(t.shape) {
			dimSize := t.shape[pos]
			if dimSize > 1 {
				index += indices[i] * strides[pos]
			}
		}
	}
	return index
}

func computeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

func canBroadcast(a, b []int) bool {
	for i := 1; i <= len(a) || i <= len(b); i++ {
		dimA := 1
		if i <= len(a) {
			dimA = a[len(a)-i]
		}
		dimB := 1
		if i <= len(b) {
			dimB = b[len(b)-i]
		}
		if dimA != dimB && dimA != 1 && dimB != 1 {
			return false
		}
	}
	return true
}

func getBroadcastedShape(a, b []int) []int {
	if len(a) == 0 {
		return b
	}
	if len(b) == 0 {
		return a
	}

	maxLen := max(len(a), len(b))
	shape := make([]int, maxLen)
	for i := range shape {
		dimA := 1
		if i < len(a) {
			dimA = a[len(a)-1-i]
		}
		dimB := 1
		if i < len(b) {
			dimB = b[len(b)-1-i]
		}
		if dimA == 1 || dimB == 1 {
			shape[maxLen-1-i] = max(dimA, dimB)
		} else if dimA == dimB {
			shape[maxLen-1-i] = dimA
		} else {
			panic("cannot broadcast shapes")
		}
	}
	return shape
}

func product(dims []int) int {
	total := 1
	for _, dim := range dims {
		total *= dim
	}
	return total
}
