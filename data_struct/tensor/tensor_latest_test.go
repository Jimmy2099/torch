package tensor

import "testing"

func TestTensor_TransposeByDim(t *testing.T) {
	// 创建测试张量 shape: [2, 3]
	td := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})

	// 转置行列
	tT := td.TransposeByDim(0, 1)

	// 验证结果
	expected := NewTensor([]float64{1, 4, 2, 5, 3, 6}, []int{3, 2})
	if !tT.Equal(expected) {
		panic("Transpose failed")
	}
}
