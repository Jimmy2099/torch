package tensor

import (
	"math"
	"testing"
)

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

func TestMaxByDim(t *testing.T) {
	// 测试二维张量
	t.Run("2D矩阵列最大值", func(t *testing.T) {
		data := []float64{
			1, 5, 3,
			4, 2, 6,
		}
		input := NewTensor(data, []int{2, 3})

		// 测试dim=1，保持维度
		max1 := input.MaxByDim(1, true)
		expectedShape := []int{2, 1}
		if !shapeEqual(max1.Shape, expectedShape) {
			t.Errorf("形状错误，期望%v，得到%v", expectedShape, max1.Shape)
		}
		expectedData := []float64{5, 6}
		if !sliceEqual(max1.Data, expectedData, 1e-6) {
			t.Errorf("数据错误，期望%v，得到%v", expectedData, max1.Data)
		}

		// 测试dim=0，不保持维度
		max0 := input.MaxByDim(0, false)
		expectedShape = []int{3}
		if !shapeEqual(max0.Shape, expectedShape) {
			t.Errorf("形状错误，期望%v，得到%v", expectedShape, max0.Shape)
		}
		expectedData = []float64{4, 5, 6}
		if !sliceEqual(max0.Data, expectedData, 1e-6) {
			t.Errorf("数据错误，期望%v，得到%v", expectedData, max0.Data)
		}
	})

	// 测试三维张量
	t.Run("3D张量深度最大值", func(t *testing.T) {
		data := []float64{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
		}
		input := NewTensor(data, []int{3, 1, 4})

		max2 := input.MaxByDim(2, true)
		expected := []float64{4, 8, 12}
		if !sliceEqual(max2.Data, expected, 1e-6) {
			t.Errorf("3D张量最大值错误，期望%v，得到%v", expected, max2.Data)
		}
	})
}

func TestGetIndices(t *testing.T) {

	tests := []struct {
		name     string
		shape    []int
		index    int
		expected []int
	}{
		{
			name:     "2x3矩阵索引3",
			shape:    []int{2, 3},
			index:    3,
			expected: []int{1, 0}, // 正确结果
		},
		{
			name:     "3x2x4张量索引10",
			shape:    []int{3, 2, 4},
			index:    10,
			expected: []int{1, 0, 2}, // 修正后的正确期望
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dummy := NewTensor(nil, tt.shape)
			result := dummy.getIndices(tt.index)
			if !intSliceEqual(result, tt.expected) {
				t.Errorf("索引转换错误，输入%d，期望%v，得到%v",
					tt.index, tt.expected, result)
			}
		})
	}
}

func TestGetBroadcastedValue(t *testing.T) {
	// 创建形状为[3,1]的掩码
	maskData := []float64{1, 0, 1}
	mask := NewTensor(maskData, []int{3, 1})

	tests := []struct {
		indices  []int
		expected float64
	}{
		{[]int{0, 0}, 1},
		{[]int{1, 1}, 0}, // 广播到第二维的0
		{[]int{2, 3}, 1}, // 广播到第二维的0
	}

	for i, tt := range tests {
		val := mask.getBroadcastedValue(tt.indices)
		if math.Abs(val-tt.expected) > 1e-6 {
			t.Errorf("用例%d错误，索引%v，期望%.1f，得到%.1f",
				i, tt.indices, tt.expected, val)
		}
	}
}

func TestSumByDim2(t *testing.T) {
	data := []float64{
		1, 2,
		3, 4,
		5, 6,
	}
	input := NewTensor(data, []int{3, 2})

	// 测试dim=0求和
	sum0 := input.SumByDim2(0, true)
	expected := []float64{9, 12} // (1+3+5), (2+4+6)
	if !sliceEqual(sum0.Data, expected, 1e-6) {
		t.Errorf("dim0求和错误，期望%v，得到%v", expected, sum0.Data)
	}

	// 测试dim=1不保持维度
	sum1 := input.SumByDim2(1, false)
	expected = []float64{3, 7, 11} // 各行求和
	if !sliceEqual(sum1.Data, expected, 1e-6) {
		t.Errorf("dim1求和错误，期望%v，得到%v", expected, sum1.Data)
	}
}

func TestMaskedFill(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	input := NewTensor(data, []int{2, 2})
	mask := NewTensor([]float64{1, 0, 0, 1}, []int{2, 2})

	filled := input.MaskedFill(mask, -math.Inf(1))
	expected := []float64{
		math.Inf(-1), 2,
		3, math.Inf(-1),
	}

	for i, v := range filled.Data {
		if !isInf(v) && !isInf(expected[i]) {
			if math.Abs(v-expected[i]) > 1e-6 {
				t.Errorf("位置%d错误，期望%v，得到%v", i, expected[i], v)
			}
		} else if !sameInf(v, expected[i]) {
			t.Errorf("位置%d的无穷值不匹配", i)
		}
	}
}

func TestSoftmaxByDim(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	input := NewTensor(data, []int{2, 2})

	// 测试dim=1
	softmax := input.SoftmaxByDim(1)
	sum0 := softmax.Data[0] + softmax.Data[1]
	sum1 := softmax.Data[2] + softmax.Data[3]

	// 验证概率和为1
	if math.Abs(sum0-1.0) > 1e-6 || math.Abs(sum1-1.0) > 1e-6 {
		t.Errorf("softmax概率和不为1: %.6f, %.6f", sum0, sum1)
	}

	// 验证数值顺序
	if softmax.Data[0] >= softmax.Data[1] {
		t.Error("第一行softmax顺序错误")
	}
	if softmax.Data[2] >= softmax.Data[3] {
		t.Error("第二行softmax顺序错误")
	}
}

func sliceEqual(a, b []float64, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tolerance {
			return false
		}
	}
	return true
}

func intSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func isInf(f float64) bool {
	return math.IsInf(f, 1) || math.IsInf(f, -1)
}

func sameInf(a, b float64) bool {
	return math.IsInf(a, 1) && math.IsInf(b, 1) ||
		math.IsInf(a, -1) && math.IsInf(b, -1)
}

func TestTensor_ShapeCopy(t *testing.T) {
	// 定义辅助断言函数
	assertShapeEqual := func(t *testing.T, got, want []int) {
		t.Helper()
		if len(got) != len(want) {
			t.Errorf("长度不匹配: got %v, want %v", got, want)
			return
		}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("索引 %d 不匹配: got %d, want %d", i, got[i], want[i])
			}
		}
	}

	// 测试用例
	t.Run("Nil Shape", func(t *testing.T) {
		// 直接构造而不是使用NewTensor来测试nil情况
		tsr := &Tensor{
			Data:  []float64{1, 2, 3},
			Shape: nil,
		}
		copyShape := tsr.ShapeCopy()
		if copyShape != nil {
			t.Errorf("期望nil，实际得到: %v", copyShape)
		}
	})

	t.Run("Empty Shape", func(t *testing.T) {
		tsr := NewTensor([]float64{}, []int{})
		copyShape := tsr.ShapeCopy()
		assertShapeEqual(t, copyShape, []int{})
	})

	t.Run("Standard 2D Shape", func(t *testing.T) {
		tsr := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
		copyShape := tsr.ShapeCopy()
		assertShapeEqual(t, copyShape, []int{2, 2})
	})

	t.Run("High Dimension Shape", func(t *testing.T) {
		tsr := NewTensor(make([]float64, 24), []int{2, 3, 4})
		copyShape := tsr.ShapeCopy()
		assertShapeEqual(t, copyShape, []int{2, 3, 4})
	})

	t.Run("Deep Copy Verification", func(t *testing.T) {
		originalShape := []int{3, 4, 5}
		tsr := NewTensor(make([]float64, 60), originalShape)
		copyShape := tsr.ShapeCopy()

		// 修改复制后的shape
		copyShape[0] = 99
		copyShape[1] = 100

		// 验证原始shape未改变
		assertShapeEqual(t, tsr.Shape, originalShape)
	})

	t.Run("Zero Value Dimensions", func(t *testing.T) {
		tsr := NewTensor([]float64{}, []int{0})
		copyShape := tsr.ShapeCopy()
		assertShapeEqual(t, copyShape, []int{0})
	})

	t.Run("Complex Shape with Zeros", func(t *testing.T) {
		tsr := NewTensor(make([]float64, 0), []int{0, 2, 0})
		copyShape := tsr.ShapeCopy()
		assertShapeEqual(t, copyShape, []int{0, 2, 0})
	})
}
