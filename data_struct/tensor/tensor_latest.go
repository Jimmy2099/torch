package tensor

import "fmt"

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

// SplitLastDim 分割最后一个维度
func (t *Tensor) SplitLastDim(splitPoint, part int) *Tensor {
	if len(t.Shape) == 0 {
		panic("无法分割空维度")
	}
	lastDim := t.Shape[len(t.Shape)-1]
	if splitPoint <= 0 || splitPoint >= lastDim {
		panic("无效分割点")
	}

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[len(newShape)-1] = splitPoint

	start := part * splitPoint
	end := start + splitPoint
	if end > lastDim {
		end = lastDim
	}

	data := make([]float64, product(newShape))
	stride := product(t.Shape[len(t.Shape)-1:])

	fmt.Println(stride)

	for i := 0; i < product(t.Shape[:len(t.Shape)-1]); i++ {
		srcStart := i*lastDim + start
		srcEnd := i*lastDim + end
		dstStart := i * splitPoint
		copy(data[dstStart:dstStart+splitPoint], t.Data[srcStart:srcEnd])
	}

	return &Tensor{
		Data:  data,
		Shape: newShape,
	}
}

// Slice 沿指定维度对张量进行切片操作
// 参数：
//
//	start - 起始索引（包含）
//	end   - 结束索引（不包含）
//	dim   - 要切片的维度（0-based）
func (t *Tensor) Slice(start, end, dim int) *Tensor {
	// 输入验证
	if t == nil {
		panic("尝试切片空张量")
	}
	ndim := len(t.Shape)
	if dim < 0 || dim >= ndim {
		panic(fmt.Sprintf("无效切片维度 %d (总维度 %d)", dim, ndim))
	}
	if start < 0 || end > t.Shape[dim] || start >= end {
		panic(fmt.Sprintf("无效切片范围 [%d:%d] 在维度 %d (长度 %d)",
			start, end, dim, t.Shape[dim]))
	}

	// 计算新形状
	newShape := make([]int, ndim)
	copy(newShape, t.Shape)
	newShape[dim] = end - start

	// 计算原张量的步幅
	strides := computeStrides(t.Shape)
	fmt.Println(strides)

	// 计算新张量数据
	newSize := product(newShape)
	newData := make([]float64, newSize)

	// 预计算维度元数据
	sliceDimSize := t.Shape[dim]
	outerDims := product(t.Shape[:dim])   // 外层维度乘积
	innerDims := product(t.Shape[dim+1:]) // 内层维度乘积

	// 高效内存访问模式
	for outer := 0; outer < outerDims; outer++ {
		// 计算外层偏移量
		outerOffset := outer * sliceDimSize * innerDims

		// 计算切片区域
		sliceStart := outerOffset + start*innerDims
		sliceEnd := outerOffset + end*innerDims

		// 拷贝切片数据
		copySize := (end - start) * innerDims
		copy(newData[outer*copySize:], t.Data[sliceStart:sliceEnd])
	}

	return &Tensor{
		Data:  newData,
		Shape: newShape,
	}
}

// Concat 沿指定维度拼接两个张量
func (t *Tensor) Concat(other *Tensor, dim int) *Tensor {
	// 输入验证
	if t == nil || other == nil {
		panic("输入张量不能为空")
	}
	if len(t.Shape) != len(other.Shape) {
		panic(fmt.Sprintf("维度数不匹配：%d vs %d",
			len(t.Shape), len(other.Shape)))
	}
	ndim := len(t.Shape)
	if dim < 0 || dim >= ndim {
		panic(fmt.Sprintf("无效维度 %d（总维度 %d）", dim, ndim))
	}

	// 验证非拼接维度一致性
	for i := 0; i < ndim; i++ {
		if i != dim && t.Shape[i] != other.Shape[i] {
			panic(fmt.Sprintf("维度 %d 大小不匹配：%d vs %d",
				i, t.Shape[i], other.Shape[i]))
		}
	}

	// 计算新形状
	newShape := make([]int, ndim)
	copy(newShape, t.Shape)
	newShape[dim] = t.Shape[dim] + other.Shape[dim]

	// 预计算步幅
	tStrides := computeStrides(t.Shape)
	otherStrides := computeStrides(other.Shape)
	newStrides := computeStrides(newShape)

	// 创建结果张量
	newData := make([]float64, product(newShape))

	// 遍历所有元素
	for pos := 0; pos < len(newData); pos++ {
		// 将线性索引转换为多维索引
		indices := make([]int, ndim)
		remaining := pos
		for d := 0; d < ndim; d++ {
			indices[d] = remaining / newStrides[d]
			remaining = remaining % newStrides[d]
		}

		// 确定数据来源
		var src *Tensor
		var srcStrides []int
		if indices[dim] < t.Shape[dim] {
			src = t
			srcStrides = tStrides
		} else {
			src = other
			srcStrides = otherStrides
			indices[dim] -= t.Shape[dim] // 调整other的索引
		}

		// 计算源位置
		srcPos := 0
		for d := 0; d < ndim; d++ {
			srcPos += indices[d] * srcStrides[d]
		}

		// 复制数据
		if srcPos < len(src.Data) {
			newData[pos] = src.Data[srcPos]
		}
	}

	return &Tensor{
		Data:  newData,
		Shape: newShape,
	}
}
