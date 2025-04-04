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

// 辅助函数：创建指定形状并填充连续数据的张量
func createTensor(shape []int) *Tensor {
	size := product(shape)
	data := make([]float64, size)
	for i := range data {
		data[i] = float64(i)
	}
	return &Tensor{Data: data, Shape: shape}
}

func TestSplitLastDim(t *testing.T) {
	t.Run("正常分割最后一个维度", func(t *testing.T) {
		tensor := createTensor([]int{2, 4}) // 数据: [0,1,2,3,4,5,6,7]
		split := tensor.SplitLastDim(2, 0)
		expected := &Tensor{
			Data:  []float64{0, 1, 4, 5},
			Shape: []int{2, 2},
		}
		if !split.Equal(expected) {
			t.Errorf("分割结果错误\n期望: %v\n实际: %v", expected, split)
		}
	})

	t.Run("分割不足部分应补零（潜在bug）", func(t *testing.T) {
		tensor := createTensor([]int{2, 5}) // 数据: 0-9
		split := tensor.SplitLastDim(3, 1)
		// 预期形状 [2,3]，实际数据应为 [3,4,0,8,9,0]
		expected := &Tensor{
			Data:  []float64{3, 4, 0, 8, 9, 0},
			Shape: []int{2, 3},
		}
		if !split.Equal(expected) {
			t.Errorf("分割不足部分错误\n期望: %v\n实际: %v", expected, split)
		}
	})

	t.Run("空张量panic", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("未触发空张量panic")
			}
		}()
		(&Tensor{Shape: []int{}}).SplitLastDim(1, 0)
	})

	t.Run("无效分割点panic", func(t *testing.T) {
		tensor := createTensor([]int{2, 3})
		defer func() {
			if r := recover(); r == nil {
				t.Error("未触发无效分割点panic")
			}
		}()
		tensor.SplitLastDim(3, 0) // splitPoint >= lastDim(3)
	})
}

func TestSlice(t *testing.T) {
	t.Run("中间维度切片", func(t *testing.T) {
		tensor := createTensor([]int{3, 4, 5})
		sliced := tensor.Slice(1, 3, 1)
		expected := &Tensor{
			Shape: []int{3, 2, 5},
			Data:  make([]float64, 3*2*5),
		}
		// 填充预期数据（手动计算切片区域）
		for i := 0; i < 3; i++ {
			copy(expected.Data[i*10:(i+1)*10], tensor.Data[i*20+5:i*20+15])
		}
		if !sliced.Equal(expected) {
			t.Error("中间维度切片错误")
		}
	})

	t.Run("末尾维度切片", func(t *testing.T) {
		tensor := createTensor([]int{2, 3})
		sliced := tensor.Slice(0, 2, 1)
		expected := &Tensor{
			Shape: []int{2, 2},
			Data:  []float64{0, 1, 3, 4},
		}

		if !sliced.Equal(expected) {
			t.Errorf("末尾切片错误\n期望: %v\n实际: %v", expected, sliced)
		}
	})

	t.Run("非法范围panic", func(t *testing.T) {
		tensor := createTensor([]int{2, 3})
		defer func() {
			if r := recover(); r == nil {
				t.Error("未触发非法范围panic")
			}
		}()
		tensor.Slice(2, 1, 0) // start > end
	})
}

func TestConcat(t *testing.T) {
	t.Run("拼接第0维度", func(t *testing.T) {
		t1 := createTensor([]int{2, 3})
		t2 := createTensor([]int{3, 3})
		concatenated := t1.Concat(t2, 0)
		expected := &Tensor{
			Shape: []int{5, 3},
			Data:  make([]float64, 5*3),
		}
		copy(expected.Data[:6], t1.Data)
		copy(expected.Data[6:], t2.Data)
		if !concatenated.Equal(expected) {
			t.Error("第0维拼接错误")
		}
	})

	t.Run("非拼接维度不匹配panic", func(t *testing.T) {
		t1 := createTensor([]int{2, 3})
		t2 := createTensor([]int{2, 4})
		defer func() {
			if r := recover(); r == nil {
				t.Error("未触发维度不匹配panic")
			}
		}()
		t1.Concat(t2, 0)
	})

	t.Run("末尾维度拼接", func(t *testing.T) {
		t1 := createTensor([]int{2, 2})
		t2 := createTensor([]int{2, 3})
		concatenated := t1.Concat(t2, 1)
		expected := &Tensor{
			Shape: []int{2, 5},
			Data: []float64{
				0, 1, 0, 1, 2,
				2, 3, 3, 4, 5,
			},
		}
		if !concatenated.Equal(expected) {
			t.Error("末尾维度拼接错误")
		}
	})
}
