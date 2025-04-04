package tensor

// Transpose 交换两个维度的位置，返回新的张量
func (t *Tensor) TransposeByDim(dim1, dim2 int) *Tensor {
	if len(t.Shape) < 2 {
		panic("Tensor must have at least 2 dimensions to transpose")
	}
	if dim1 < 0 || dim1 >= len(t.Shape) || dim2 < 0 || dim2 >= len(t.Shape) {
		panic("Invalid transpose dimensions")
	}

	// 创建新的形状
	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[dim1], newShape[dim2] = newShape[dim2], newShape[dim1]

	// 计算步幅
	oldStrides := computeStrides(t.Shape)
	newStrides := computeStrides(newShape)

	// 创建新数据数组
	newData := make([]float64, len(t.Data))
	for i := range newData {
		// 将线性索引转换为多维索引
		indices := make([]int, len(t.Shape))
		remaining := i
		for d := 0; d < len(newShape); d++ {
			indices[d] = remaining / newStrides[d]
			remaining = remaining % newStrides[d]
		}

		// 交换维度索引
		indices[dim1], indices[dim2] = indices[dim2], indices[dim1]

		// 计算原线性索引
		oldIndex := 0
		for d := 0; d < len(t.Shape); d++ {
			oldIndex += indices[d] * oldStrides[d]
		}

		newData[i] = t.Data[oldIndex]
	}

	return &Tensor{
		Data:  newData,
		Shape: newShape,
	}
}
