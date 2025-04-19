package tensor

import (
	math "github.com/chewxy/math32"
	"reflect"
	"slices"
	"testing"
)

func TestTensor_TransposeByDim(t *testing.T) {
	td := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})

	tT := td.TransposeByDim(0, 1)

	expected := NewTensor([]float32{1, 4, 2, 5, 3, 6}, []int{3, 2})
	if !tT.Equal(expected) {
		panic("Transpose failed")
	}
}

func createTensor(shape []int) *Tensor {
	size := product(shape)
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i)
	}
	return &Tensor{Data: data, shape: shape}
}

func TestSplitLastDim(t *testing.T) {
	t.Run("Split the last dimension normally", func(t *testing.T) {
		tensor := createTensor([]int{2, 4})
		split := tensor.SplitLastDim(2, 0)
		expected := &Tensor{
			Data:  []float32{0, 1, 4, 5},
			shape: []int{2, 2},
		}
		if !split.Equal(expected) {
			t.Errorf("Split result error\nExpected: %v\nActual: %v", expected, split)
		}
	})

	t.Run("Insufficient split should be padded with zeros (potential bug)", func(t *testing.T) {
		tensor := createTensor([]int{2, 5})
		split := tensor.SplitLastDim(3, 1)
		expected := &Tensor{
			Data:  []float32{3, 4, 0, 8, 9, 0},
			shape: []int{2, 3},
		}
		if !split.Equal(expected) {
			t.Errorf("Insufficient split error\nExpected: %v\nActual: %v", expected, split)
		}
	})

	t.Run("Empty tensor panic", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Empty tensor panic not triggered")
			}
		}()
		(&Tensor{shape: []int{}}).SplitLastDim(1, 0)
	})

	t.Run("Invalid split point panic", func(t *testing.T) {
		tensor := createTensor([]int{2, 3})
		defer func() {
			if r := recover(); r == nil {
				t.Error("Invalid split point panic not triggered")
			}
		}()
		tensor.SplitLastDim(3, 0)
	})
}

func TestSlice(t *testing.T) {
	t.Run("Slice the middle dimension", func(t *testing.T) {
		tensor := createTensor([]int{3, 4, 5})
		sliced := tensor.Slice(1, 3, 1)
		expected := &Tensor{
			shape: []int{3, 2, 5},
			Data:  make([]float32, 3*2*5),
		}
		for i := 0; i < 3; i++ {
			copy(expected.Data[i*10:(i+1)*10], tensor.Data[i*20+5:i*20+15])
		}
		if !sliced.Equal(expected) {
			t.Error("Error slicing the middle dimension")
		}
	})

	t.Run("Slice the last dimension", func(t *testing.T) {
		tensor := createTensor([]int{2, 3})
		sliced := tensor.Slice(0, 2, 1)
		expected := &Tensor{
			shape: []int{2, 2},
			Data:  []float32{0, 1, 3, 4},
		}

		if !sliced.Equal(expected) {
			t.Errorf("Last slice error\nExpected: %v\nActual: %v", expected, sliced)
		}
	})

	t.Run("Illegal range panic", func(t *testing.T) {
		tensor := createTensor([]int{2, 3})
		defer func() {
			if r := recover(); r == nil {
				t.Error("Illegal range panic not triggered")
			}
		}()
		tensor.Slice(2, 1, 0)
	})
}

func TestConcat(t *testing.T) {
	t.Run("Concatenate the 0th dimension", func(t *testing.T) {
		t1 := createTensor([]int{2, 3})
		t2 := createTensor([]int{3, 3})
		concatenated := t1.Concat(t2, 0)
		expected := &Tensor{
			shape: []int{5, 3},
			Data:  make([]float32, 5*3),
		}
		copy(expected.Data[:6], t1.Data)
		copy(expected.Data[6:], t2.Data)
		if !concatenated.Equal(expected) {
			t.Error("Error concatenating the 0th dimension")
		}
	})

	t.Run("Non-concatenation dimension mismatch panic", func(t *testing.T) {
		t1 := createTensor([]int{2, 3})
		t2 := createTensor([]int{2, 4})
		defer func() {
			if r := recover(); r == nil {
				t.Error("Dimension mismatch panic not triggered")
			}
		}()
		t1.Concat(t2, 0)
	})

	t.Run("Concatenate the last dimension", func(t *testing.T) {
		t1 := createTensor([]int{2, 2})
		t2 := createTensor([]int{2, 3})
		concatenated := t1.Concat(t2, 1)
		expected := &Tensor{
			shape: []int{2, 5},
			Data: []float32{
				0, 1, 0, 1, 2,
				2, 3, 3, 4, 5,
			},
		}
		if !concatenated.Equal(expected) {
			t.Error("Error concatenating the last dimension")
		}
	})

}

func TestMaxByDim(t *testing.T) {
	t.Run("2D matrix column maximum", func(t *testing.T) {
		data := []float32{
			1, 5, 3,
			4, 2, 6,
		}
		input := NewTensor(data, []int{2, 3})

		max1 := input.MaxByDim(1, true)
		expectedShape := []int{2, 1}
		if !shapeEqual(max1.shape, expectedShape) {
			t.Errorf("shape error, expected %v, got %v", expectedShape, max1.shape)
		}
		expectedData := []float32{5, 6}
		if !sliceEqual(max1.Data, expectedData, 1e-6) {
			t.Errorf("Data error, expected %v, got %v", expectedData, max1.Data)
		}

		max0 := input.MaxByDim(0, false)
		expectedShape = []int{3}
		if !shapeEqual(max0.shape, expectedShape) {
			t.Errorf("shape error, expected %v, got %v", expectedShape, max0.shape)
		}
		expectedData = []float32{4, 5, 6}
		if !sliceEqual(max0.Data, expectedData, 1e-6) {
			t.Errorf("Data error, expected %v, got %v", expectedData, max0.Data)
		}
	})

	t.Run("3D tensor depth maximum", func(t *testing.T) {
		data := []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
		}
		input := NewTensor(data, []int{3, 1, 4})

		max2 := input.MaxByDim(2, true)
		expected := []float32{4, 8, 12}
		if !sliceEqual(max2.Data, expected, 1e-6) {
			t.Errorf("3D tensor maximum error, expected %v, got %v", expected, max2.Data)
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
			name:     "2x3 matrix index 3",
			shape:    []int{2, 3},
			index:    3,
			expected: []int{1, 0},
		},
		{
			name:     "3x2x4 tensor index 10",
			shape:    []int{3, 2, 4},
			index:    10,
			expected: []int{1, 0, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dummy := NewTensor(nil, tt.shape)
			result := dummy.getIndices(tt.index)
			if !intSliceEqual(result, tt.expected) {
				t.Errorf("Index conversion error, input %d, expected %v, got %v",
					tt.index, tt.expected, result)
			}
		})
	}
}

func TestGetBroadcastedValue(t *testing.T) {
	maskData := []float32{1, 0, 1}
	mask := NewTensor(maskData, []int{3, 1})

	tests := []struct {
		indices  []int
		expected float32
	}{
		{[]int{0, 0}, 1},
		{[]int{1, 1}, 0},
		{[]int{2, 3}, 1},
	}

	for i, tt := range tests {
		val := mask.getBroadcastedValue(tt.indices)
		if math.Abs(val-tt.expected) > 1e-6 {
			t.Errorf("Case %d error, index %v, expected %.1f, got %.1f",
				i, tt.indices, tt.expected, val)
		}
	}
}

func TestSumByDim2(t *testing.T) {
	data := []float32{
		1, 2,
		3, 4,
		5, 6,
	}
	input := NewTensor(data, []int{3, 2})

	sum0 := input.SumByDim2(0, true)
	expected := []float32{9, 12}
	if !sliceEqual(sum0.Data, expected, 1e-6) {
		t.Errorf("dim0 summation error, expected %v, got %v", expected, sum0.Data)
	}

	sum1 := input.SumByDim2(1, false)
	expected = []float32{3, 7, 11}
	if !sliceEqual(sum1.Data, expected, 1e-6) {
		t.Errorf("dim1 summation error, expected %v, got %v", expected, sum1.Data)
	}
}

func TestMaskedFill(t *testing.T) {
	data := []float32{1, 2, 3, 4}
	input := NewTensor(data, []int{2, 2})
	mask := NewTensor([]float32{1, 0, 0, 1}, []int{2, 2})

	filled := input.MaskedFill(mask, -math.Inf(1))
	expected := []float32{
		math.Inf(-1), 2,
		3, math.Inf(-1),
	}

	for i, v := range filled.Data {
		if !isInf(v) && !isInf(expected[i]) {
			if math.Abs(v-expected[i]) > 1e-6 {
				t.Errorf("Position %d error, expected %v, got %v", i, expected[i], v)
			}
		} else if !sameInf(v, expected[i]) {
			t.Errorf("Infinite value at position %d does not match", i)
		}
	}
}

func TestSoftmaxByDim(t *testing.T) {
	data := []float32{1, 2, 3, 4}
	input := NewTensor(data, []int{2, 2})

	softmax := input.SoftmaxByDim(1)
	sum0 := softmax.Data[0] + softmax.Data[1]
	sum1 := softmax.Data[2] + softmax.Data[3]

	if math.Abs(sum0-1.0) > 1e-6 || math.Abs(sum1-1.0) > 1e-6 {
		t.Errorf("Softmax probability sum is not 1: %.6f, %.6f", sum0, sum1)
	}

	if softmax.Data[0] >= softmax.Data[1] {
		t.Error("First row softmax order error")
	}
	if softmax.Data[2] >= softmax.Data[3] {
		t.Error("Second row softmax order error")
	}
}

func sliceEqual(a, b []float32, tolerance float32) bool {
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

func isInf(f float32) bool {
	return math.IsInf(f, 1) || math.IsInf(f, -1)
}

func sameInf(a, b float32) bool {
	return math.IsInf(a, 1) && math.IsInf(b, 1) ||
		math.IsInf(a, -1) && math.IsInf(b, -1)
}

func TestTensor_ShapeCopy(t *testing.T) {
	assertShapeEqual := func(t *testing.T, got, want []int) {
		t.Helper()
		if len(got) != len(want) {
			t.Errorf("Length mismatch: got %v, want %v", got, want)
			return
		}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("Index %d mismatch: got %d, want %d", i, got[i], want[i])
			}
		}
	}

	t.Run("Nil shape", func(t *testing.T) {
		tsr := &Tensor{
			Data:  []float32{1, 2, 3},
			shape: nil,
		}
		copyShape := tsr.ShapeCopy()
		if copyShape != nil {
			t.Errorf("Expected nil, but got: %v", copyShape)
		}
	})

	t.Run("Empty shape", func(t *testing.T) {
		tsr := NewTensor([]float32{}, []int{})
		copyShape := tsr.ShapeCopy()
		assertShapeEqual(t, copyShape, []int{})
	})

	t.Run("Standard 2D shape", func(t *testing.T) {
		tsr := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
		copyShape := tsr.ShapeCopy()
		assertShapeEqual(t, copyShape, []int{2, 2})
	})

	t.Run("High Dimension shape", func(t *testing.T) {
		tsr := NewTensor(make([]float32, 24), []int{2, 3, 4})
		copyShape := tsr.ShapeCopy()
		assertShapeEqual(t, copyShape, []int{2, 3, 4})
	})

	t.Run("Deep Copy Verification", func(t *testing.T) {
		originalShape := []int{3, 4, 5}
		tsr := NewTensor(make([]float32, 60), originalShape)
		copyShape := tsr.ShapeCopy()

		copyShape[0] = 99
		copyShape[1] = 100

		assertShapeEqual(t, tsr.shape, originalShape)
	})

	t.Run("Zero Value Dimensions", func(t *testing.T) {
		tsr := NewTensor([]float32{}, []int{0})
		copyShape := tsr.ShapeCopy()
		assertShapeEqual(t, copyShape, []int{0})
	})

	t.Run("Complex shape with Zeros", func(t *testing.T) {
		tsr := NewTensor(make([]float32, 0), []int{0, 2, 0})
		copyShape := tsr.ShapeCopy()
		assertShapeEqual(t, copyShape, []int{0, 2, 0})
	})
}

func Test2DMatrixMultiplication(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})

	expected := []float32{
		1*5 + 2*7, 1*6 + 2*8,
		3*5 + 4*7, 3*6 + 4*8,
	}

	result := a.MatMul(b)

	if !slices.Equal(result.Data, expected) {
		t.Errorf("Result error:\nExpected: %v\nActual: %v",
			expected,
			result.Data)
	}
}

func TestRepeatInterleave(t *testing.T) {
	tests := []struct {
		name     string
		input    *Tensor
		dim      int
		repeats  int
		expected *Tensor
	}{
		{
			name: "2D dim0 repeats2",
			input: NewTensor(
				[]float32{1, 2, 3, 4, 5, 6},
				[]int{2, 3},
			),
			dim:     0,
			repeats: 2,
			expected: NewTensor(
				[]float32{
					1, 2, 3,
					1, 2, 3,
					4, 5, 6,
					4, 5, 6,
				},
				[]int{4, 3},
			),
		},
		{
			name: "2D dim1 repeats3",
			input: NewTensor(
				[]float32{1, 2, 3, 4},
				[]int{2, 2},
			),
			dim:     1,
			repeats: 3,
			expected: NewTensor(
				[]float32{
					1, 1, 1, 2, 2, 2,
					3, 3, 3, 4, 4, 4,
				},
				[]int{2, 6},
			),
		},
		{
			name: "2D dim1 repeats1 (no change)",
			input: NewTensor(
				[]float32{1, 2, 3, 4},
				[]int{2, 2},
			),
			dim:     1,
			repeats: 1,
			expected: NewTensor(
				[]float32{1, 2, 3, 4},
				[]int{2, 2},
			),
		},

		{
			name: "4D dim1 repeats2 channels",
			input: NewTensor(
				[]float32{
					1, 2, 3, 4,
					5, 6, 7, 8,
				},
				[]int{1, 2, 2, 2},
			),
			dim:     1,
			repeats: 2,
			expected: NewTensor(
				[]float32{
					1, 2, 3, 4,
					1, 2, 3, 4,
					5, 6, 7, 8,
					5, 6, 7, 8,
				},
				[]int{1, 4, 2, 2},
			),
		},
		{
			name: "4D dim0 repeats3 batch",
			input: NewTensor(
				[]float32{
					1, 1, 1, 1,
					2, 2, 2, 2,
				},
				[]int{2, 1, 2, 2},
			),
			dim:     0,
			repeats: 3,
			expected: NewTensor(
				[]float32{
					1, 1, 1, 1,
					1, 1, 1, 1,
					1, 1, 1, 1,
					2, 2, 2, 2,
					2, 2, 2, 2,
					2, 2, 2, 2,
				},
				[]int{6, 1, 2, 2},
			),
		},
		{
			name: "4D multi-batch channels repeat",
			input: NewTensor(
				[]float32{
					1, 1, 2, 2,
					3, 3, 4, 4,
					5, 5, 6, 6,
					7, 7, 8, 8,
				},
				[]int{2, 2, 2, 2},
			),
			dim:     1,
			repeats: 2,
			expected: NewTensor(
				[]float32{
					1, 1, 2, 2,
					1, 1, 2, 2,
					3, 3, 4, 4,
					3, 3, 4, 4,
					5, 5, 6, 6,
					5, 5, 6, 6,
					7, 7, 8, 8,
					7, 7, 8, 8,
				},
				[]int{2, 4, 2, 2},
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.input.RepeatInterleave(tt.dim, tt.repeats)

			if !result.Equal(tt.expected) {
				t.Errorf("Data mismatch\nExpected: %v %v Actual: %v %v", tt.expected.Data[:5], result.Data[:5], tt.expected.shape, result.shape)
			}
		})
	}
}

func TestGetValue(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		indices   []int
		expected  float32
		wantPanic bool
	}{
		{
			name:      "1D valid index",
			tensor:    NewTensor([]float32{1, 2, 3, 4}, []int{4}),
			indices:   []int{2},
			expected:  3,
			wantPanic: false,
		},
		{
			name:      "1D index out of range",
			tensor:    NewTensor([]float32{1, 2}, []int{2}),
			indices:   []int{2},
			wantPanic: true,
		},

		{
			name:      "2D valid index",
			tensor:    NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}),
			indices:   []int{1, 2},
			expected:  6,
			wantPanic: false,
		},
		{
			name:      "2D negative index",
			tensor:    NewTensor([]float32{1, 2, 3}, []int{3, 1}),
			indices:   []int{-1, 0},
			wantPanic: true,
		},

		{
			name:      "3D valid index",
			tensor:    NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 2}),
			indices:   []int{1, 0, 1},
			expected:  6,
			wantPanic: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); (r != nil) != tt.wantPanic {
					t.Errorf("GetValue() panic = %v, wantPanic %v", r, tt.wantPanic)
				}
			}()

			got := tt.tensor.GetValue(tt.indices)
			if got != tt.expected {
				t.Errorf("GetValue() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestMaskedFill1(t *testing.T) {
	tests := []struct {
		name     string
		input    *Tensor
		mask     *Tensor
		value    float32
		expected []float32
	}{
		{
			name:     "No broadcast - exact shape match",
			input:    NewTensor([]float32{1, 2, 3, 4}, []int{2, 2}),
			mask:     NewTensor([]float32{0, 1, 1, 0}, []int{2, 2}),
			value:    -99,
			expected: []float32{1, -99, -99, 4},
		},

		{
			name:  "Broadcast with size-1 dimensions",
			input: NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}),
			mask:  NewTensor([]float32{0, 1}, []int{1, 2}),
			value: -99,
			expected: []float32{
				1, -99,
				3, -99,
				5, -99,
			},
		},

		{
			name:     "Full masking",
			input:    NewTensor([]float32{1, 2, 3}, []int{3}),
			mask:     NewTensor([]float32{1, 1, 1}, []int{3}),
			value:    0,
			expected: []float32{0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.input.MaskedFill(tt.mask, tt.value)
			if !reflect.DeepEqual(result.Data, tt.expected) {
				t.Errorf("MaskedFill() mismatch:\nGot: %v\nWant: %v", result.Data, tt.expected)
			}
		})
	}
}

func TestTensorRoundTo(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		decimals int
		expected []float32
	}{
		{
			name:     "Keep 0 decimal places (integer)",
			input:    []float32{3.1415, -2.7182, 9.9999},
			decimals: 0,
			expected: []float32{3, -3, 10},
		},
		{
			name:     "Keep 3 decimal places",
			input:    []float32{1.2345678, 5.4321098, -0.0004999},
			decimals: 3,
			expected: []float32{1.235, 5.432, -0.000},
		},
		{
			name:     "Keep 8 decimal places (maximum precision)",
			input:    []float32{0.123456789, 0.000000001},
			decimals: 8,
			expected: []float32{0.12345679, 0.00000000},
		},
		{
			name:     "Illegal number of decimal places (automatically corrected to 0)",
			input:    []float32{2.5},
			decimals: -2,
			expected: []float32{3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(tt.input, []int{len(tt.input)})
			rounded := tensor.RoundTo(tt.decimals)

			const epsilon = 1e-7
			for i, v := range rounded.Data {
				if diff := math.Abs(float32(v - tt.expected[i])); diff > epsilon {
					t.Errorf("Index %d: Input %.8f → Expected %.8f, Actual %.8f (Error %.8f)",
						i, tt.input[i], tt.expected[i], v, diff)
				}
			}
		})
	}
}
