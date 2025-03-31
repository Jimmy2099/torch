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
