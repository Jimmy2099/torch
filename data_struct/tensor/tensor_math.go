package tensor

import "fmt"

func (t *Tensor) Sub(other *Tensor) *Tensor {
	if false {
		if t.ShapesMatch(other) {
			return t.Sub_bak(other)
		}
	}

	{
		// 检查空张量
		if len(t.Data) == 0 || len(other.Data) == 0 {
			// 返回与输入形状相同的空张量
			emptyShape := getBroadcastedShape(t.Shape, other.Shape)
			return &Tensor{
				Shape: emptyShape,
				Data:  make([]float64, 0),
			}
		}
		// 检查是否可广播
		if !canBroadcast(t.Shape, other.Shape) {
			panic(fmt.Sprintf("无法广播形状 %v 和 %v", t.Shape, other.Shape))
		}
	}

	{
		// 获取广播后的形状
		broadcastedShape := getBroadcastedShape(t.Shape, other.Shape)
		result := Zeros(broadcastedShape) // 假设存在创建零张量的函数

		// 计算每个张量的strides
		tStrides := computeStrides(t.Shape)
		otherStrides := computeStrides(other.Shape)

		// 遍历每个元素的位置
		size := result.Size()
		for i := 0; i < size; i++ {
			indices := result.Indices(i)
			tIndex := t.broadcastedIndex(indices, tStrides)
			otherIndex := other.broadcastedIndex(indices, otherStrides)
			result.Data[i] = t.Data[tIndex] - other.Data[otherIndex]
		}

		return result
	}
}

func (t *Tensor) Div(other *Tensor) *Tensor {
	if false {
		if t.ShapesMatch(other) {
			return t.Div_bak(other)
		}
	}

	{
		// 检查空张量
		if len(t.Data) == 0 || len(other.Data) == 0 {
			// 返回与输入形状相同的空张量
			emptyShape := getBroadcastedShape(t.Shape, other.Shape)
			return &Tensor{
				Shape: emptyShape,
				Data:  make([]float64, 0),
			}
		}
		// 检查是否可广播
		if !canBroadcast(t.Shape, other.Shape) {
			panic(fmt.Sprintf("无法广播形状 %v 和 %v", t.Shape, other.Shape))
		}
	}

	{
		// 获取广播后的形状
		broadcastedShape := getBroadcastedShape(t.Shape, other.Shape)
		result := Zeros(broadcastedShape) // 假设存在创建零张量的函数

		// 计算每个张量的strides
		tStrides := computeStrides(t.Shape)
		otherStrides := computeStrides(other.Shape)

		// 遍历每个元素的位置
		size := result.Size()
		for i := 0; i < size; i++ {
			indices := result.Indices(i)
			tIndex := t.broadcastedIndex(indices, tStrides)
			otherIndex := other.broadcastedIndex(indices, otherStrides)
			result.Data[i] = t.Data[tIndex] / other.Data[otherIndex]
		}

		return result
	}
}

// Add 张量加法（支持广播）
func (t *Tensor) Add(other *Tensor) *Tensor {
	{
		// 检查空张量
		if len(t.Data) == 0 || len(other.Data) == 0 {
			// 返回与输入形状相同的空张量
			emptyShape := getBroadcastedShape(t.Shape, other.Shape)
			return &Tensor{
				Shape: emptyShape,
				Data:  make([]float64, 0),
			}
		}
		// 检查是否可广播
		if !canBroadcast(t.Shape, other.Shape) {
			panic(fmt.Sprintf("无法广播形状 %v 和 %v", t.Shape, other.Shape))
		}
	}

	// 获取广播后的形状
	broadcastedShape := getBroadcastedShape(t.Shape, other.Shape)
	result := Zeros(broadcastedShape) // 假设存在创建零张量的函数

	// 计算每个张量的strides
	tStrides := computeStrides(t.Shape)
	otherStrides := computeStrides(other.Shape)

	// 遍历每个元素的位置
	size := result.Size()
	for i := 0; i < size; i++ {
		indices := result.Indices(i)
		tIndex := t.broadcastedIndex(indices, tStrides)
		otherIndex := other.broadcastedIndex(indices, otherStrides)
		result.Data[i] = t.Data[tIndex] + other.Data[otherIndex] // 改为加法运算
	}

	return result
}

// Mul 张量乘法（支持广播）
func (t *Tensor) Mul(other *Tensor) *Tensor {
	{
		// 检查空张量
		if len(t.Data) == 0 || len(other.Data) == 0 {
			// 返回与输入形状相同的空张量
			emptyShape := getBroadcastedShape(t.Shape, other.Shape)
			return &Tensor{
				Shape: emptyShape,
				Data:  make([]float64, 0),
			}
		}
		// 检查是否可广播
		if !canBroadcast(t.Shape, other.Shape) {
			panic(fmt.Sprintf("无法广播形状 %v 和 %v", t.Shape, other.Shape))
		}
	}
	{

		// 获取广播后的形状
		broadcastedShape := getBroadcastedShape(t.Shape, other.Shape)
		result := Zeros(broadcastedShape) // 假设存在创建零张量的函数

		// 计算每个张量的strides
		tStrides := computeStrides(t.Shape)
		otherStrides := computeStrides(other.Shape)

		// 遍历每个元素的位置
		size := result.Size()
		for i := 0; i < size; i++ {
			indices := result.Indices(i)
			tIndex := t.broadcastedIndex(indices, tStrides)
			otherIndex := other.broadcastedIndex(indices, otherStrides)
			result.Data[i] = t.Data[tIndex] * other.Data[otherIndex] // 改为乘法运算
		}

		return result
	}
}

// MatMul 矩阵乘法 支持批量矩阵乘法
func (t *Tensor) MatMul1(other *Tensor) *Tensor {
	return t.MatMul(other)
}

func (t *Tensor) MatMul(other *Tensor) *Tensor {
	a := t
	b := other
	aIs1D := false
	bIs1D := false

	// 处理一维张量
	if len(a.Shape) == 1 {
		a = a.Reshape(append([]int{1}, a.Shape...))
		aIs1D = true
	}
	if len(b.Shape) == 1 {
		b = b.Reshape(append(b.Shape, 1))
		bIs1D = true
	}

	// 处理空张量
	if len(a.Data) == 0 || len(b.Data) == 0 {
		aBatch := a.Shape[:len(a.Shape)-2]
		bBatch := b.Shape[:len(b.Shape)-2]
		batchShape := getBroadcastedShape(aBatch, bBatch)
		m := a.Shape[len(a.Shape)-2]
		p := b.Shape[len(b.Shape)-1]
		resultShape := append(append([]int{}, batchShape...), m, p)
		if aIs1D {
			resultShape = resultShape[:len(resultShape)-1]
		}
		if bIs1D {
			resultShape = resultShape[:len(resultShape)-1]
		}
		return &Tensor{
			Shape: resultShape,
			Data:  make([]float64, 0),
		}
	}

	// 检查最后两个维度是否兼容
	aLastDim := a.Shape[len(a.Shape)-1]
	bSecondLastDim := b.Shape[len(b.Shape)-2]
	if aLastDim != bSecondLastDim {
		panic(fmt.Sprintf("matmul: dimension mismatch (%d vs %d)", aLastDim, bSecondLastDim))
	}

	// 获取批量维度并计算广播后的形状
	aBatchShape := a.Shape[:len(a.Shape)-2]
	bBatchShape := b.Shape[:len(b.Shape)-2]
	batchShape := getBroadcastedShape(aBatchShape, bBatchShape)
	if batchShape == nil {
		panic(fmt.Sprintf("matmul: cannot broadcast batch shapes %v and %v", aBatchShape, bBatchShape))
	}

	m := a.Shape[len(a.Shape)-2]
	p := b.Shape[len(b.Shape)-1]
	resultShape := append(append([]int{}, batchShape...), m, p)
	result := Zeros(resultShape)

	aStrides := computeStrides(a.Shape)
	bStrides := computeStrides(b.Shape)
	batchSize := product(batchShape)
	matrixSize := m * p

	for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
		indices := getBatchIndices(batchIdx, batchShape)
		aBatchIndices := getBroadcastedIndices(indices, aBatchShape, batchShape)
		bBatchIndices := getBroadcastedIndices(indices, bBatchShape, batchShape)

		aOffset := 0
		for i, idx := range aBatchIndices {
			aOffset += idx * aStrides[i]
		}
		bOffset := 0
		for i, idx := range bBatchIndices {
			bOffset += idx * bStrides[i]
		}

		aMatrix := a.Data[aOffset : aOffset+m*aLastDim]
		bMatrix := b.Data[bOffset : bOffset+bSecondLastDim*p]
		resultMatrix := matrixMultiply(aMatrix, bMatrix, m, aLastDim, p)

		resultOffset := batchIdx * matrixSize
		copy(result.Data[resultOffset:resultOffset+matrixSize], resultMatrix)
	}

	// 压缩一维情况
	if aIs1D {
		newShape := append(result.Shape[:len(result.Shape)-2], result.Shape[len(result.Shape)-1])
		result = result.Reshape(newShape)
	}
	if bIs1D {
		result = result.Reshape(result.Shape[:len(result.Shape)-1])
	}

	return result
}

// 辅助函数
func getBatchIndices(batchIdx int, shape []int) []int {
	indices := make([]int, len(shape))
	for i := range shape {
		size := product(shape[i+1:])
		indices[i] = batchIdx / size
		batchIdx %= size
	}
	return indices
}

func getBroadcastedIndices(bcIndices, origShape, bcShape []int) []int {
	offset := len(bcShape) - len(origShape)
	indices := make([]int, len(origShape))
	for i := range origShape {
		bcDim := offset + i
		if origShape[i] == 1 {
			indices[i] = 0
		} else {
			indices[i] = bcIndices[bcDim]
		}
	}
	return indices
}

func matrixMultiply(a, b []float64, m, n, p int) []float64 {
	result := make([]float64, m*p)
	for i := 0; i < m; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += a[i*n+k] * b[k*p+j]
			}
			result[i*p+j] = sum
		}
	}
	return result
}
