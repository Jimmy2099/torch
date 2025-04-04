package tensor

import (
	"reflect"
	"testing"
	"time"
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

func TestTensorAdd(t *testing.T) {
	// 测试不同形状的张量
	tests := []struct {
		aShape, bShape []int
	}{
		{[]int{256}, []int{1, 256, 1, 1}}, // 原始测试
		{[]int{3, 1}, []int{1, 4}},        // 经典广播案例
		{[]int{5, 4}, []int{1}},           // 标量广播
	}

	for _, tt := range tests {
		a := Ones(tt.aShape)
		b := Zeros(tt.bShape)
		result := a.Add(b)
		if result == nil {
			t.Errorf("Add returned nil for shapes %v and %v", tt.aShape, tt.bShape)
		}
	}
}

func TestTensorAdd1(t *testing.T) {
	// 基础测试：相同形状
	t.Run("SameShape", func(t *testing.T) {
		a := Ones([]int{2, 3})
		b := Ones([]int{2, 3})
		result := a.Add(b)
		expected := []float64{2, 2, 2, 2, 2, 2}
		if !reflect.DeepEqual(result.TensorData(), expected) {
			t.Errorf("Expected %v, got %v", expected, result.TensorData())
		}
	})

	// 经典广播：尾部维度对齐
	t.Run("TrailingBroadcast", func(t *testing.T) {
		a := Ones([]int{4, 3})
		b := Ones([]int{3})
		result := a.Add(b)
		expected := []float64{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
		if !reflect.DeepEqual(result.TensorData(), expected) {
			t.Errorf("Expected %v, got %v", expected, result.TensorData())
		}
	})

	// 中间维度广播
	t.Run("MiddleDimBroadcast", func(t *testing.T) {
		a := Ones([]int{2, 1, 4})
		b := Ones([]int{2, 3, 4})
		result := a.Add(b)
		if result.Shape[1] != 3 { // 检查广播后的形状
			t.Errorf("Expected broadcasted shape [2 3 4], got %v", result.Shape)
		}
	})

	// 高维广播
	t.Run("HighDimBroadcast", func(t *testing.T) {
		a := Ones([]int{1, 256, 1, 1}) // 原始问题中的形状
		b := Ones([]int{256})
		result := a.Add(b)
		if !reflect.DeepEqual(result.Shape, []int{1, 256, 1, 256}) {
			t.Errorf("Shape mismatch: expected [1 256 1 256], got %v", result.Shape)
		}
	})

	// 标量广播
	t.Run("ScalarBroadcast", func(t *testing.T) {
		a := Ones([]int{2, 2})
		b := NewTensor([]float64{5}, []int{1}) // 标量
		result := a.Add(b)
		expected := []float64{6, 6, 6, 6}
		if !reflect.DeepEqual(result.TensorData(), expected) {
			t.Errorf("Expected %v, got %v", expected, result.TensorData())
		}
	})

	// 不兼容形状（应panic）
	t.Run("IncompatibleShapes", func(t *testing.T) {
		a := Ones([]int{3, 4})
		b := Ones([]int{2, 3})
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for incompatible shapes")
			}
		}()
		a.Add(b)
	})

	// 零维张量
	t.Run("EmptyTensor", func(t *testing.T) {
		testCases := []struct {
			aShape, bShape []int
		}{
			{[]int{0}, []int{1}},       // 空张量 + 标量
			{[]int{1}, []int{0}},       // 标量 + 空张量
			{[]int{0, 2}, []int{1, 2}}, // 多维空张量
		}

		for _, tc := range testCases {
			a := Ones(tc.aShape)
			b := Ones(tc.bShape)
			result := a.Add(b)

			// 预期结果应该是空张量
			if len(result.TensorData()) != 0 {
				t.Errorf("Expected empty result for shapes %v + %v", tc.aShape, tc.bShape)
			}

			// 检查输出形状是否正确
			expectedShape := getBroadcastedShape(tc.aShape, tc.bShape)
			if !reflect.DeepEqual(result.Shape, expectedShape) {
				t.Errorf("Shape mismatch: expected %v, got %v", expectedShape, result.Shape)
			}
		}
	})

	// 性能测试：大张量
	t.Run("LargeTensor", func(t *testing.T) {
		if testing.Short() {
			t.Skip("Skipping large tensor test in short mode")
		}
		a := Ones([]int{1000, 1000})
		b := Ones([]int{1000, 1}) // 列向量
		start := time.Now()
		result := a.Add(b)
		elapsed := time.Since(start)
		t.Logf("Large tensor add took %v", elapsed)
		if result.Shape[0] != 1000 || result.Shape[1] != 1000 {
			t.Error("Shape mismatch in large tensor add")
		}
	})

	// 3D广播测试
	t.Run("3DBroadcast", func(t *testing.T) {
		a := Ones([]int{3, 1, 5})
		b := Ones([]int{1, 4, 1})
		result := a.Add(b)

		// 预期广播后的形状应该是 [3,4,5]
		expectedShape := []int{3, 4, 5}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("3D广播形状错误: 预期 %v, 得到 %v", expectedShape, result.Shape)
		}

		// 检查所有元素值应为2
		for _, val := range result.TensorData() {
			if val != 2 {
				t.Error("3D广播结果值错误: 预期所有元素为2")
				break
			}
		}
	})

	// 4D广播测试
	t.Run("4DBroadcast", func(t *testing.T) {
		a := Ones([]int{1, 16, 1, 8})
		b := Ones([]int{16, 1, 8})
		result := a.Add(b)

		// 预期广播后的形状应该是 [1,16,1,8]
		expectedShape := []int{1, 16, 1, 8}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("4D广播形状错误: 预期 %v, 得到 %v", expectedShape, result.Shape)
		}

		// 检查所有元素值应为2
		for _, val := range result.TensorData() {
			if val != 2 {
				t.Error("4D广播结果值错误: 预期所有元素为2")
				break
			}
		}
	})
}

func TestMatMul99(t *testing.T) {
	// ----------------------------
	// 测试用例 1: 基本二维矩阵乘法
	// ----------------------------
	t.Run("Basic 2D MatMul", func(t *testing.T) {
		a := NewTensor(
			[]float64{1, 2, 3, 4}, // [[1 2], [3 4]]
			[]int{2, 2},
		)
		b := NewTensor(
			[]float64{5, 6, 7, 8}, // [[5 6], [7 8]]
			[]int{2, 2},
		)
		result := a.MatMul(b)
		expected := []float64{
			1*5 + 2*7, // [0][0] = (1,2) · (5,7)
			1*6 + 2*8, // [0][1] = (1,2) · (6,8)
			3*5 + 4*7, // [1][0] = (3,4) · (5,7)
			3*6 + 4*8, // [1][1] = (3,4) · (6,8)
		}
		assertTensorEqual(t, result, expected, []int{2, 2})
	})

	// ----------------------------
	// 测试用例 2: 批量矩阵乘法
	// ----------------------------
	t.Run("Batch MatMul", func(t *testing.T) {
		a := NewTensor(
			[]float64{
				// Batch 0
				1, 2, // [[1 2]
				3, 4, //  [3 4]]
				// Batch 1
				5, 6, // [[5 6]
				7, 8}, //  [7 8]]
			[]int{2, 2, 2},
		)
		b := NewTensor(
			[]float64{
				// Batch 0
				9, 10, // [[9 10]
				11, 12, //  [11 12]]
				// Batch 1
				13, 14, // [[13 14]
				15, 16}, //  [15 16]]
			[]int{2, 2, 2},
		)
		result := a.MatMul(b)
		expected := []float64{
			// Batch 0
			1*9 + 2*11,  // [0][0][0]
			1*10 + 2*12, // [0][0][1]
			3*9 + 4*11,  // [0][1][0]
			3*10 + 4*12, // [0][1][1]

			// Batch 1
			5*13 + 6*15, // [1][0][0]
			5*14 + 6*16, // [1][0][1]
			7*13 + 8*15, // [1][1][0]
			7*14 + 8*16, // [1][1][1]
		}
		assertTensorEqual(t, result, expected, []int{2, 2, 2})
	})

	// ----------------------------
	// 测试用例 3: 广播批量维度
	// ----------------------------
	t.Run("Broadcast Batch Dims", func(t *testing.T) {
		a := NewTensor(
			[]float64{1, 2, 3, 4, 5, 6}, // 形状 [3,2]，广播为 [2,3,2]
			[]int{3, 2},
		)
		b := NewTensor(
			[]float64{
				7, 8, // Batch 0 [[7], [8]]
				11, 12, // Batch 1 [[11], [12]]
			},
			[]int{2, 2, 1}, // 正确形状，4个元素
		)
		result := a.MatMul(b)
		expected := []float64{
			// Batch 0
			1*7 + 2*8, // 23
			3*7 + 4*8, // 53
			5*7 + 6*8, // 83

			// Batch 1
			1*11 + 2*12, // 35
			3*11 + 4*12, // 81
			5*11 + 6*12, // 127
		}
		assertTensorEqual(t, result, expected, []int{2, 3, 1})
	})

	// ----------------------------
	// 测试用例 4: 一维张量（向量）
	// ----------------------------
	t.Run("1D Vectors", func(t *testing.T) {
		vec := NewTensor([]float64{1, 2, 3}, []int{3}) // 被广播为 (1,3)
		mat := NewTensor(
			[]float64{
				4, 5, // 第0行
				6, 7, // 第1行
				8, 9, // 第2行
			},
			[]int{3, 2},
		)
		result := vec.MatMul(mat)
		expected := []float64{
			1*4 + 2*6 + 3*8, // (1,2,3) · 第0列(4,6,8)
			1*5 + 2*7 + 3*9, // (1,2,3) · 第1列(5,7,9)
		}
		assertTensorEqual(t, result, expected, []int{2})
	})

}

// 辅助断言函数
func assertTensorEqual(t *testing.T, actual *Tensor, expectedData []float64, expectedShape []int) {
	t.Helper()
	if !shapeEqual(actual.Shape, expectedShape) {
		t.Errorf("形状不匹配\n期望: %v\n实际: %v", expectedShape, actual.Shape)
	}
	if len(actual.Data) != len(expectedData) {
		t.Fatalf("数据长度不匹配: 期望 %d 实际 %d", len(expectedData), len(actual.Data))
	}
	for i := range actual.Data {
		if actual.Data[i] != expectedData[i] {
			t.Errorf("数据不匹配在索引 %d\n期望: %v\n实际: %v", i, expectedData[i], actual.Data[i])
			break
		}
	}
}
