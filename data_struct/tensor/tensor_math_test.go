package tensor

import (
	"reflect"
	"testing"
)

func TestTensorSub(t *testing.T) {
	// 测试相同形状的张量
	a := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	b := NewTensor([]float64{4, 3, 2, 1}, []int{2, 2})
	expected := NewTensor([]float64{-3, -1, 1, 3}, []int{2, 2})

	result := a.Sub(b)
	if !reflect.DeepEqual(result.Data, expected.Data) {
		t.Errorf("Sub 计算错误: 期望 %v, 但得到 %v", expected.Data, result.Data)
	}

	// 测试广播（形状 [2,2] 和 [1,2]）
	c := NewTensor([]float64{1, 2}, []int{1, 2})
	expectedBroadcast := NewTensor([]float64{0, 0, 2, 2}, []int{2, 2})

	resultBroadcast := a.Sub(c)
	if !reflect.DeepEqual(resultBroadcast.Data, expectedBroadcast.Data) {
		t.Errorf("广播减法计算错误: 期望 %v, 但得到 %v", expectedBroadcast.Data, resultBroadcast.Data)
	}

	// 测试广播（形状 [2,2] 和 [2,1]）
	d := NewTensor([]float64{1, 2}, []int{2, 1})
	expectedBroadcast2 := NewTensor([]float64{0, 1, 1, 2}, []int{2, 2})

	resultBroadcast2 := a.Sub(d)
	if !reflect.DeepEqual(resultBroadcast2.Data, expectedBroadcast2.Data) {
		t.Errorf("广播减法计算错误: 期望 %v, 但得到 %v", expectedBroadcast2.Data, resultBroadcast2.Data)
	}
}
