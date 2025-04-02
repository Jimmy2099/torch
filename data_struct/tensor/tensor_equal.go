package tensor

import (
	"math"
)

func (t *Tensor) Equal(other *Tensor) bool {
	// 1. 空指针检查
	if t == nil || other == nil {
		return t == nil && other == nil
	}

	// 2. 形状比较
	if !shapeEqual(t.Shape, other.Shape) {
		return false
	}

	// 3. 数据长度验证
	if len(t.Data) != len(other.Data) {
		return false
	}

	// 4. 元素级浮点数比较
	for i := range t.Data {
		if t.Data[i] != other.Data[i] {
			return false
		}
	}

	return true
}

// shapeEqual 检查两个张量形状是否相同
func shapeEqual(shape1, shape2 []int) bool {
	if len(shape1) != len(shape2) {
		return false
	}
	for i := range shape1 {
		if shape1[i] != shape2[i] {
			return false
		}
	}
	return true
}

// 可选：带容差的比较方法
func (t *Tensor) EqualWithTolerance(other *Tensor, epsilon float64) bool {
	if t == nil || other == nil {
		return t == nil && other == nil
	}

	if !shapeEqual(t.Shape, other.Shape) {
		return false
	}

	for i := range t.Data {
		if math.Abs(t.Data[i]-other.Data[i]) > epsilon {
			return false
		}
	}

	return true
}
