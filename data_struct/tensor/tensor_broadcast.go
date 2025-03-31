package tensor

// 辅助函数：计算多维索引对应的原始张量索引
func (t *Tensor) broadcastedIndex(indices []int, strides []int) int {
	index := 0
	for i := 0; i < len(indices); i++ {
		dimSize := t.Shape[i]
		if dimSize > 1 {
			index += indices[i] * strides[i]
		}
		// 如果dimSize == 1，则忽略，因为使用0
	}
	return index
}

// computeStrides 计算张量的步幅
func computeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// canBroadcast 检查两个形状是否可广播
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

// getBroadcastedShape 获取广播后的形状
func getBroadcastedShape(a, b []int) []int {
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
			panic("无法广播形状")
		}
	}
	return shape
}
