package tensor

import (
	"reflect"
	"testing"
	"time"
)

func TestTensorSub(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensor([]float32{4, 3, 2, 1}, []int{2, 2})
	expected := NewTensor([]float32{-3, -1, 1, 3}, []int{2, 2})

	result := a.Sub(b)
	if !reflect.DeepEqual(result.Data, expected.Data) {
		t.Errorf("Sub 计算错误: 期望 %v, 但得到 %v", expected.Data, result.Data)
	}

	c := NewTensor([]float32{1, 2}, []int{1, 2})
	expectedBroadcast := NewTensor([]float32{0, 0, 2, 2}, []int{2, 2})

	resultBroadcast := a.Sub(c)
	if !reflect.DeepEqual(resultBroadcast.Data, expectedBroadcast.Data) {
		t.Errorf("广播减法计算错误: 期望 %v, 但得到 %v", expectedBroadcast.Data, resultBroadcast.Data)
	}

	d := NewTensor([]float32{1, 2}, []int{2, 1})
	expectedBroadcast2 := NewTensor([]float32{0, 1, 1, 2}, []int{2, 2})

	resultBroadcast2 := a.Sub(d)
	if !reflect.DeepEqual(resultBroadcast2.Data, expectedBroadcast2.Data) {
		t.Errorf("广播减法计算错误: 期望 %v, 但得到 %v", expectedBroadcast2.Data, resultBroadcast2.Data)
	}
}

func TestTensorAdd(t *testing.T) {
	tests := []struct {
		aShape, bShape []int
	}{
		{[]int{256}, []int{1, 256, 1, 1}},
		{[]int{3, 1}, []int{1, 4}},
		{[]int{5, 4}, []int{1}},
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
	t.Run("SameShape", func(t *testing.T) {
		a := Ones([]int{2, 3})
		b := Ones([]int{2, 3})
		result := a.Add(b)
		expected := []float32{2, 2, 2, 2, 2, 2}
		if !reflect.DeepEqual(result.TensorData(), expected) {
			t.Errorf("Expected %v, got %v", expected, result.TensorData())
		}
	})

	t.Run("TrailingBroadcast", func(t *testing.T) {
		a := Ones([]int{4, 3})
		b := Ones([]int{3})
		result := a.Add(b)
		expected := []float32{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
		if !reflect.DeepEqual(result.TensorData(), expected) {
			t.Errorf("Expected %v, got %v", expected, result.TensorData())
		}
	})

	t.Run("MiddleDimBroadcast", func(t *testing.T) {
		a := Ones([]int{2, 1, 4})
		b := Ones([]int{2, 3, 4})
		result := a.Add(b)
		if result.Shape[1] != 3 {
			t.Errorf("Expected broadcasted shape [2 3 4], got %v", result.Shape)
		}
	})

	t.Run("HighDimBroadcast", func(t *testing.T) {
		a := Ones([]int{1, 256, 1, 1})
		b := Ones([]int{256})
		result := a.Add(b)
		if !reflect.DeepEqual(result.Shape, []int{1, 256, 1, 256}) {
			t.Errorf("Shape mismatch: expected [1 256 1 256], got %v", result.Shape)
		}
	})

	t.Run("ScalarBroadcast", func(t *testing.T) {
		a := Ones([]int{2, 2})
		b := NewTensor([]float32{5}, []int{1})
		result := a.Add(b)
		expected := []float32{6, 6, 6, 6}
		if !reflect.DeepEqual(result.TensorData(), expected) {
			t.Errorf("Expected %v, got %v", expected, result.TensorData())
		}
	})

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

	t.Run("EmptyTensor", func(t *testing.T) {
		testCases := []struct {
			aShape, bShape []int
		}{
			{[]int{0}, []int{1}},
			{[]int{1}, []int{0}},
			{[]int{0, 2}, []int{1, 2}},
		}

		for _, tc := range testCases {
			a := Ones(tc.aShape)
			b := Ones(tc.bShape)
			result := a.Add(b)

			if len(result.TensorData()) != 0 {
				t.Errorf("Expected empty result for shapes %v + %v", tc.aShape, tc.bShape)
			}

			expectedShape := getBroadcastedShape(tc.aShape, tc.bShape)
			if !reflect.DeepEqual(result.Shape, expectedShape) {
				t.Errorf("Shape mismatch: expected %v, got %v", expectedShape, result.Shape)
			}
		}
	})

	t.Run("LargeTensor", func(t *testing.T) {
		if testing.Short() {
			t.Skip("Skipping large tensor test in short mode")
		}
		a := Ones([]int{1000, 1000})
		b := Ones([]int{1000, 1})
		start := time.Now()
		result := a.Add(b)
		elapsed := time.Since(start)
		t.Logf("Large tensor add took %v", elapsed)
		if result.Shape[0] != 1000 || result.Shape[1] != 1000 {
			t.Error("Shape mismatch in large tensor add")
		}
	})

	t.Run("3DBroadcast", func(t *testing.T) {
		a := Ones([]int{3, 1, 5})
		b := Ones([]int{1, 4, 1})
		result := a.Add(b)

		expectedShape := []int{3, 4, 5}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("3D广播形状错误: 预期 %v, 得到 %v", expectedShape, result.Shape)
		}

		for _, val := range result.TensorData() {
			if val != 2 {
				t.Error("3D广播结果值错误: 预期所有元素为2")
				break
			}
		}
	})

	t.Run("4DBroadcast", func(t *testing.T) {
		a := Ones([]int{1, 16, 1, 8})
		b := Ones([]int{16, 1, 8})
		result := a.Add(b)

		expectedShape := []int{1, 16, 1, 8}
		if !reflect.DeepEqual(result.Shape, expectedShape) {
			t.Errorf("4D广播形状错误: 预期 %v, 得到 %v", expectedShape, result.Shape)
		}

		for _, val := range result.TensorData() {
			if val != 2 {
				t.Error("4D广播结果值错误: 预期所有元素为2")
				break
			}
		}
	})
}

func TestMatMul99(t *testing.T) {
	t.Run("Basic 2D MatMul", func(t *testing.T) {
		a := NewTensor(
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
		)
		b := NewTensor(
			[]float32{5, 6, 7, 8},
			[]int{2, 2},
		)
		result := a.MatMul(b)
		expected := []float32{
			1*5 + 2*7,
			1*6 + 2*8,
			3*5 + 4*7,
			3*6 + 4*8,
		}
		assertTensorEqual(t, result, expected, []int{2, 2})
	})

	t.Run("Batch MatMul", func(t *testing.T) {
		a := NewTensor(
			[]float32{
				1, 2,
				3, 4,
				5, 6,
				7, 8},
			[]int{2, 2, 2},
		)
		b := NewTensor(
			[]float32{
				9, 10,
				11, 12,
				13, 14,
				15, 16},
			[]int{2, 2, 2},
		)
		result := a.MatMul(b)
		expected := []float32{
			1*9 + 2*11,
			1*10 + 2*12,
			3*9 + 4*11,
			3*10 + 4*12,

			5*13 + 6*15,
			5*14 + 6*16,
			7*13 + 8*15,
			7*14 + 8*16,
		}
		assertTensorEqual(t, result, expected, []int{2, 2, 2})
	})

	t.Run("Broadcast Batch Dims", func(t *testing.T) {
		a := NewTensor(
			[]float32{1, 2, 3, 4, 5, 6},
			[]int{3, 2},
		)
		b := NewTensor(
			[]float32{
				7, 8,
				11, 12,
			},
			[]int{2, 2, 1},
		)
		result := a.MatMul(b)
		expected := []float32{
			1*7 + 2*8,
			3*7 + 4*8,
			5*7 + 6*8,

			1*11 + 2*12,
			3*11 + 4*12,
			5*11 + 6*12,
		}
		assertTensorEqual(t, result, expected, []int{2, 3, 1})
	})

	t.Run("1D Vectors", func(t *testing.T) {
		vec := NewTensor([]float32{1, 2, 3}, []int{3})
		mat := NewTensor(
			[]float32{
				4, 5,
				6, 7,
				8, 9,
			},
			[]int{3, 2},
		)
		result := vec.MatMul(mat)
		expected := []float32{
			1*4 + 2*6 + 3*8,
			1*5 + 2*7 + 3*9,
		}
		assertTensorEqual(t, result, expected, []int{2})
	})

}

func assertTensorEqual(t *testing.T, actual *Tensor, expectedData []float32, expectedShape []int) {
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
