package tensor

// Copy 创建张量的深拷贝
func Copy(t *Tensor) *Tensor {
	data := make([]float64, len(t.Data))
	copy(data, t.Data)
	return NewTensor(data, append([]int{}, t.Shape...))
}

// Size 返回张量中元素的总数
func (t *Tensor) Size() int {
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// At 获取指定位置的元素（适用于任意维度）
func (t *Tensor) At(indices ...int) float64 {
	if len(indices) != len(t.Shape) {
		panic("number of indices must match tensor rank")
	}

	pos := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.Shape[i] {
			panic("tensor: index out of range")
		}
		pos += indices[i] * stride
		stride *= t.Shape[i]
	}

	return t.Data[pos]
}
