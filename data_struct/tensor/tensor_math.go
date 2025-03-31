package tensor

import "fmt"

// Sub 张量减法（支持广播）
func (t *Tensor) Sub(other *Tensor) *Tensor {
	// 检查是否可广播
	if !canBroadcast(t.Shape, other.Shape) {
		panic(fmt.Sprintf("无法广播形状 %v 和 %v", t.Shape, other.Shape))
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
		result.Data[i] = t.Data[tIndex] - other.Data[otherIndex]
	}

	return result
}
