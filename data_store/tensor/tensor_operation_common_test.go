package tensor

import (
	math "github.com/chewxy/math32"
	"reflect"
	"testing"
)

const epsilon = 1e-9

func tensorsEqual(t1, t2 *Tensor, tol float32) bool {
	if t1 == nil || t2 == nil {
		return t1 == t2
	}
	if !reflect.DeepEqual(t1.shape, t2.shape) {
		return false
	}
	return floatsEqual(t1.Data, t2.Data, tol)
}

func checkPanic1(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()
	f()
}

func TestTensor_Size(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		want  int
	}{
		{"Scalar (implicit)", []int{1}, 1},
		{"Vector", []int{5}, 5},
		{"Matrix", []int{2, 3}, 6},
		{"3D Tensor", []int{2, 1, 4}, 8},
		{"Empty Tensor", []int{0}, 0},
		{"Empty Tensor High Dim", []int{2, 0, 3}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := make([]float32, tt.want)
			if tt.want == 0 && len(tt.shape) > 0 {
				zeroFound := false
				for _, d := range tt.shape {
					if d == 0 {
						zeroFound = true
						break
					}
				}
				if !zeroFound {
					t.Fatalf("Test setup error: shape %v results in size 0 but has no zero dimension", tt.shape)
				}
				tensor := &Tensor{Data: []float32{}, shape: tt.shape}
				if got := tensor.Size(); got != tt.want {
					t.Errorf("Size() = %v, want %v", got, tt.want)
				}
				return
			} else if tt.want == 0 && len(tt.shape) == 0 {
				tensor := &Tensor{}
				if got := tensor.Size(); got != 0 {
				}
				tensor = NewTensor([]float32{}, []int{})
				if got := tensor.Size(); got != 1 {
				}
				return

			}

			tensor := NewTensor(data, tt.shape)
			if got := tensor.Size(); got != tt.want {
				t.Errorf("Size() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTensor_At(t *testing.T) {
	tensor := NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,
	}, []int{2, 3})

	tests := []struct {
		name    string
		indices []int
		want    float32
		wantErr bool
	}{
		{"Valid Index (0,0)", []int{0, 0}, 1.0, false},
		{"Valid Index (0,2)", []int{0, 2}, 3.0, false},
		{"Valid Index (1,0)", []int{1, 0}, 4.0, false},
		{"Valid Index (1,2)", []int{1, 2}, 6.0, false},
		{"Invalid Index (Out of bounds 0)", []int{2, 0}, 0.0, true},
		{"Invalid Index (Out of bounds 1)", []int{0, 3}, 0.0, true},
		{"Invalid Index (Negative)", []int{0, -1}, 0.0, true},
		{"Wrong number of indices (too few)", []int{1}, 0.0, true},
		{"Wrong number of indices (too many)", []int{1, 1, 1}, 0.0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.wantErr {
				checkPanic(t, func() { tensor.At(tt.indices...) }, "")
			} else {
				got := tensor.At(tt.indices...)
				if math.Abs(got-tt.want) > epsilon {
					t.Errorf("At(%v) = %v, want %v", tt.indices, got, tt.want)
				}
			}
		})
	}

	tensor1D := NewTensor([]float32{10, 20, 30}, []int{3})
	if got := tensor1D.At(1); got != 20.0 {
		t.Errorf("At(1) on 1D tensor = %v, want %v", got, 20.0)
	}
	checkPanic(t, func() { tensor1D.At(3) }, "")
	checkPanic(t, func() { tensor1D.At(0, 0) }, "")
}

func TestOnes(t *testing.T) {
	shape := []int{2, 3}
	tensor := Ones(shape)
	wantData := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	wantTensor := NewTensor(wantData, shape)

	if !tensorsEqual(tensor, wantTensor, epsilon) {
		t.Errorf("Ones(%v) = %v, want %v", shape, tensor, wantTensor)
	}
}

func TestZeros(t *testing.T) {
	shape := []int{3, 1, 2}
	tensor := Zeros(shape)
	wantData := []float32{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	wantTensor := NewTensor(wantData, shape)

	if !tensorsEqual(tensor, wantTensor, epsilon) {
		t.Errorf("Zeros(%v) = %v, want %v", shape, tensor, wantTensor)
	}
}

func TestZerosLike(t *testing.T) {
	source := NewTensor([]float32{1, 2, 3}, []int{3})
	tensor := ZerosLike(source)
	wantData := []float32{0.0, 0.0, 0.0}
	wantTensor := NewTensor(wantData, []int{3})

	if !tensorsEqual(tensor, wantTensor, epsilon) {
		t.Errorf("ZerosLike(%v) = %v, want %v", source, tensor, wantTensor)
	}

	t.Run("PanicOnNilInput", func(t *testing.T) {
		checkPanic(t, func() { ZerosLike(nil) }, "")
	})

	t.Run("PanicOnNilShape", func(t *testing.T) {
		badTensor := &Tensor{Data: []float32{1}}
		checkPanic(t, func() { ZerosLike(badTensor) }, "")
	})
}

func TestTensor_AddScalar(t *testing.T) {
	original := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	scalar := float32(5.0)
	result := original.AddScalar(scalar)
	wantData := []float32{6, 7, 8, 9}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("AddScalar(%v) = %v, want %v", scalar, result, wantTensor)
	}

	originalWant := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	if !tensorsEqual(original, originalWant, epsilon) {
		t.Errorf("Original tensor was modified by AddScalar. Got %v, want %v", original, originalWant)
	}
}

func TestTensor_MulScalar(t *testing.T) {
	original := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	scalar := float32(3.0)
	result := original.MulScalar(scalar)
	wantData := []float32{3, 6, 9, 12}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("MulScalar(%v) = %v, want %v", scalar, result, wantTensor)
	}

	originalWant := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	if !tensorsEqual(original, originalWant, epsilon) {
		t.Errorf("Original tensor was modified by MulScalar. Got %v, want %v", original, originalWant)
	}
}

func TestTensor_Div(t *testing.T) {
	t1 := NewTensor([]float32{10, 20, 30, 40}, []int{2, 2})
	t2 := NewTensor([]float32{2, 4, 5, 8}, []int{2, 2})
	result := t1.Div(t2)
	wantData := []float32{5, 5, 6, 5}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("Div(%v / %v) = %v, want %v", t1, t2, result, wantTensor)
	}

	t1Want := NewTensor([]float32{10, 20, 30, 40}, []int{2, 2})
	t2Want := NewTensor([]float32{2, 4, 5, 8}, []int{2, 2})
	if !tensorsEqual(t1, t1Want, epsilon) {
		t.Errorf("Original tensor t1 was modified by Div. Got %v", t1)
	}
	if !tensorsEqual(t2, t2Want, epsilon) {
		t.Errorf("Original tensor t2 was modified by Div. Got %v", t2)
	}

}

func TestTensor_Sqrt(t *testing.T) {
	original := NewTensor([]float32{1, 4, 9, 16}, []int{2, 2})
	result := original.Sqrt()
	wantData := []float32{1, 2, 3, 4}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("Sqrt() = %v, want %v", result, wantTensor)
	}

	originalWant := NewTensor([]float32{1, 4, 9, 16}, []int{2, 2})
	if !tensorsEqual(original, originalWant, epsilon) {
		t.Errorf("Original tensor was modified by Sqrt. Got %v, want %v", original, originalWant)
	}

	zeroTensor := NewTensor([]float32{0.0, 4.0}, []int{2})
	sqrtZero := zeroTensor.Sqrt()
	wantSqrtZero := NewTensor([]float32{0.0, 2.0}, []int{2})
	if !tensorsEqual(sqrtZero, wantSqrtZero, epsilon) {
		t.Errorf("Sqrt() with zero = %v, want %v", sqrtZero, wantSqrtZero)
	}
	negTensor := NewTensor([]float32{-4.0, 9.0}, []int{2})
	sqrtNeg := negTensor.Sqrt()
	if !math.IsNaN(sqrtNeg.Data[0]) || math.Abs(sqrtNeg.Data[1]-3.0) > epsilon {
		t.Errorf("Sqrt() with negative input = %v, want [NaN 3.0]", sqrtNeg.Data)
	}
}

func TestTensor_Pow(t *testing.T) {
	original := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	exponent := float32(3.0)
	result := original.Pow(exponent)
	wantData := []float32{1, 8, 27, 64}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("Pow(%v) = %v, want %v", exponent, result, wantTensor)
	}

	originalWant := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	if !tensorsEqual(original, originalWant, epsilon) {
		t.Errorf("Original tensor was modified by Pow. Got %v, want %v", original, originalWant)
	}

	sqrtTensor := NewTensor([]float32{4.0, 9.0}, []int{2})
	sqrtResult := sqrtTensor.Pow(0.5)
	wantSqrtResult := NewTensor([]float32{2.0, 3.0}, []int{2})
	if !tensorsEqual(sqrtResult, wantSqrtResult, epsilon) {
		t.Errorf("Pow(0.5) = %v, want %v", sqrtResult, wantSqrtResult)
	}
}

func TestTensor_SumByDim1(t *testing.T) {
	tensor := NewTensor([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, []int{2, 2, 4})

	tests := []struct {
		name      string
		dims      []int
		keepDims  bool
		wantData  []float32
		wantShape []int
	}{
		{
			name:      "Sum dim 0, keep",
			dims:      []int{0},
			keepDims:  true,
			wantData:  []float32{10, 12, 14, 16, 18, 20, 22, 24},
			wantShape: []int{1, 2, 4},
		},
		{
			name:      "Sum dim 0, no keep",
			dims:      []int{0},
			keepDims:  false,
			wantData:  []float32{10, 12, 14, 16, 18, 20, 22, 24},
			wantShape: []int{2, 4},
		},
		{
			name:      "Sum dim 1, keep",
			dims:      []int{1},
			keepDims:  true,
			wantData:  []float32{6, 8, 10, 12, 22, 24, 26, 28},
			wantShape: []int{2, 1, 4},
		},
		{
			name:      "Sum dim 1, no keep",
			dims:      []int{1},
			keepDims:  false,
			wantData:  []float32{6, 8, 10, 12, 22, 24, 26, 28},
			wantShape: []int{2, 4},
		},
		{
			name:      "Sum dim 2, keep",
			dims:      []int{2},
			keepDims:  true,
			wantData:  []float32{10, 26, 42, 58},
			wantShape: []int{2, 2, 1},
		},
		{
			name:      "Sum dim 2, no keep",
			dims:      []int{2},
			keepDims:  false,
			wantData:  []float32{10, 26, 42, 58},
			wantShape: []int{2, 2},
		},
		{
			name:      "Sum dims 0, 2, keep",
			dims:      []int{0, 2},
			keepDims:  true,
			wantData:  []float32{52, 84},
			wantShape: []int{1, 2, 1},
		},
		{
			name:      "Sum dims 0, 2, no keep",
			dims:      []int{0, 2},
			keepDims:  false,
			wantData:  []float32{52, 84},
			wantShape: []int{2},
		},
		{
			name:      "Sum all dims, keep",
			dims:      []int{0, 1, 2},
			keepDims:  true,
			wantData:  []float32{136},
			wantShape: []int{1, 1, 1},
		},
		{
			name:      "Sum all dims, no keep",
			dims:      []int{0, 1, 2},
			keepDims:  false,
			wantData:  []float32{136},
			wantShape: []int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tensor.SumByDim1(tt.dims, tt.keepDims)
			wantTensor := NewTensor(tt.wantData, tt.wantShape)
			if len(tt.wantShape) == 0 && len(tt.wantData) == 1 {
				wantTensor = &Tensor{Data: tt.wantData, shape: []int{}}
			}

			if !tensorsEqual(result, wantTensor, epsilon) {
				t.Errorf("SumByDim1(%v, %v) = %v, want %v", tt.dims, tt.keepDims, result, wantTensor)
			}
		})
	}

	t.Run("PanicInvalidDim", func(t *testing.T) {
		checkPanic(t, func() { tensor.SumByDim1([]int{0, 3}, false) }, "")
		checkPanic(t, func() { tensor.SumByDim1([]int{-1}, false) }, "")
	})
}

func TestTensor_Mul(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	t2 := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	result := t1.Mul(t2)
	wantData := []float32{5, 12, 21, 32}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("Mul(%v * %v) = %v, want %v", t1, t2, result, wantTensor)
	}

	t1Want := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	t2Want := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	if !tensorsEqual(t1, t1Want, epsilon) {
		t.Errorf("Original tensor t1 was modified by Mul. Got %v", t1)
	}
	if !tensorsEqual(t2, t2Want, epsilon) {
		t.Errorf("Original tensor t2 was modified by Mul. Got %v", t2)
	}

}

func TestTensor_Add(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	t2 := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	result := t1.Add(t2)
	wantData := []float32{6, 8, 10, 12}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("Add(%v + %v) = %v, want %v", t1, t2, result, wantTensor)
	}

	t1Want := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	t2Want := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	if !tensorsEqual(t1, t1Want, epsilon) {
		t.Errorf("Original tensor t1 was modified by Add. Got %v", t1)
	}
	if !tensorsEqual(t2, t2Want, epsilon) {
		t.Errorf("Original tensor t2 was modified by Add. Got %v", t2)
	}

}

func TestTensor_DivScalar(t *testing.T) {
	original := NewTensor([]float32{10, 20, 30, 40}, []int{2, 2})
	scalar := float32(5.0)
	result := original.DivScalar(scalar)
	wantData := []float32{2, 4, 6, 8}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("DivScalar(%v) = %v, want %v", scalar, result, wantTensor)
	}

	originalWant := NewTensor([]float32{10, 20, 30, 40}, []int{2, 2})
	if !tensorsEqual(original, originalWant, epsilon) {
		t.Errorf("Original tensor was modified by DivScalar. Got %v, want %v", original, originalWant)
	}

	divZeroResult := original.DivScalar(0.0)
	if !math.IsInf(divZeroResult.Data[0], 1) ||
		!math.IsInf(divZeroResult.Data[1], 1) ||
		!math.IsInf(divZeroResult.Data[2], 1) ||
		!math.IsInf(divZeroResult.Data[3], 1) {
		t.Errorf("DivScalar(0.0) did not result in +Inf for all elements: %v", divZeroResult.Data)
	}

	zeroNumerator := NewTensor([]float32{0, 0, 0, 0}, []int{2, 2})
	zeroDivZeroResult := zeroNumerator.DivScalar(0.0)
	if !math.IsNaN(zeroDivZeroResult.Data[0]) ||
		!math.IsNaN(zeroDivZeroResult.Data[1]) ||
		!math.IsNaN(zeroDivZeroResult.Data[2]) ||
		!math.IsNaN(zeroDivZeroResult.Data[3]) {
		t.Errorf("DivScalar(0.0) with zero numerator did not result in NaN for all elements: %v", zeroDivZeroResult.Data)
	}
}

func TestTensor_Sum111(t *testing.T) {
	tensor := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	result := tensor.Sum111()
	wantSum := float32(1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0)
	wantTensor := NewTensor([]float32{wantSum}, []int{1})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("Sum111() = %v, want %v", result, wantTensor)
	}

	emptyTensor := NewTensor([]float32{}, []int{0})
	emptyResult := emptyTensor.Sum111()
	wantEmpty := NewTensor([]float32{0.0}, []int{1})
	if !tensorsEqual(emptyResult, wantEmpty, epsilon) {
		t.Errorf("Sum111() on empty tensor = %v, want %v", emptyResult, wantEmpty)
	}
}

func TestTensor_Get(t *testing.T) {
	tensor := NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,
	}, []int{2, 3})

	tests := []struct {
		name    string
		indices []int
		want    float32
	}{
		{"Valid Index [0, 0]", []int{0, 0}, 1.0},
		{"Valid Index [0, 2]", []int{0, 2}, 3.0},
		{"Valid Index [1, 0]", []int{1, 0}, 4.0},
		{"Valid Index [1, 2]", []int{1, 2}, 6.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tensor.Get(tt.indices)
			if math.Abs(got-tt.want) > epsilon {
				t.Errorf("Get(%v) = %v, want %v", tt.indices, got, tt.want)
			}
		})
	}

	tensor1D := NewTensor([]float32{10, 20, 30}, []int{3})
	if got := tensor1D.Get([]int{1}); got != 20.0 {
		t.Errorf("Get([1]) on 1D tensor = %v, want %v", got, 20.0)
	}

	t.Run("PanicOutOfBounds", func(t *testing.T) {
		checkPanic(t, func() { tensor.Get([]int{2, 0}) }, "")
		checkPanic(t, func() { tensor.Get([]int{0, 3}) }, "")
	})

}

func TestTensor_Set1(t *testing.T) {
	tensor := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	indices := []int{1, 0}
	value := float32(99.0)

	tensor.Set1(indices, value)

	got := tensor.Get(indices)
	if math.Abs(got-value) > epsilon {
		t.Errorf("Set1(%v, %v); Get(%v) = %v, want %v", indices, value, indices, got, value)
	}

	if math.Abs(tensor.Get([]int{0, 0})-1.0) > epsilon {
		t.Errorf("Set1 affected other element [0,0], got %v, want 1.0", tensor.Get([]int{0, 0}))
	}
	if math.Abs(tensor.Get([]int{0, 1})-2.0) > epsilon {
		t.Errorf("Set1 affected other element [0,1], got %v, want 2.0", tensor.Get([]int{0, 1}))
	}
	if math.Abs(tensor.Get([]int{1, 1})-4.0) > epsilon {
		t.Errorf("Set1 affected other element [1,1], got %v, want 4.0", tensor.Get([]int{1, 1}))
	}

	t.Run("PanicOutOfBounds", func(t *testing.T) {
		checkPanic(t, func() { tensor.Set1([]int{2, 0}, 100.0) }, "")
		checkPanic(t, func() { tensor.Set1([]int{0, 2}, 100.0) }, "")
	})
}

func TestTensor_Max1(t *testing.T) {
	tests := []struct {
		name  string
		data  []float32
		shape []int
		want  float32
	}{
		{"All Positive", []float32{1, 5, 2, 4}, []int{2, 2}, 5.0},
		{"Mixed Signs", []float32{-1, 0, -5, 3}, []int{4}, 3.0},
		{"All Negative", []float32{-10, -2, -5}, []int{3}, -2.0},
		{"Single Element", []float32{42}, []int{1}, 42.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(tt.data, tt.shape)
			got := tensor.Max1()
			if math.Abs(got-tt.want) > epsilon {
				t.Errorf("Max1() = %v, want %v", got, tt.want)
			}
		})
	}

	t.Run("Empty Tensor", func(t *testing.T) {
		emptyTensor := &Tensor{Data: []float32{}, shape: []int{0}}
		got := emptyTensor.Max1()
		if got != 0.0 {
			t.Errorf("Max1() for empty tensor = %v, want 0.0", got)
		}
	})
}

func TestTensor_Sub1(t *testing.T) {
	t1 := NewTensor([]float32{10, 20, 30, 40}, []int{2, 2})
	t2 := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	result := t1.Sub1(t2)
	wantData := []float32{9, 18, 27, 36}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("Sub1(%v - %v) = %v, want %v", t1, t2, result, wantTensor)
	}

	t1Want := NewTensor([]float32{10, 20, 30, 40}, []int{2, 2})
	t2Want := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	if !tensorsEqual(t1, t1Want, epsilon) {
		t.Errorf("Original tensor t1 was modified by Sub1. Got %v", t1)
	}
	if !tensorsEqual(t2, t2Want, epsilon) {
		t.Errorf("Original tensor t2 was modified by Sub1. Got %v", t2)
	}

	t.Run("PanicShapeMismatch", func(t *testing.T) {
		t3_diff_shape := NewTensor([]float32{1, 2, 3}, []int{3})
		t4_diff_rank := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
		t5_same_rank_diff_dim := NewTensor([]float32{1, 2, 3, 4}, []int{4, 1})

		checkPanic(t, func() { t1.Sub1(t3_diff_shape) }, "")
		checkPanic(t, func() { t1.Sub1(t4_diff_rank) }, "")
		checkPanic(t, func() { t1.Sub1(t5_same_rank_diff_dim) }, "")
	})

	t.Run("ZerosAndNegatives", func(t *testing.T) {
		a := NewTensor([]float32{1, -2, 0, 5}, []int{4})
		b := NewTensor([]float32{3, 0, -4, 5}, []int{4})
		res := a.Sub1(b)
		want := NewTensor([]float32{-2, -2, 4, 0}, []int{4})
		if !tensorsEqual(res, want, epsilon) {
			t.Errorf("Sub1 with zeros/negatives: got %v, want %v", res, want)
		}
	})
}

func TestTensor_Sum1(t *testing.T) {
	tests := []struct {
		name  string
		data  []float32
		shape []int
		want  float32
	}{
		{"All Positive", []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, 21.0},
		{"Mixed Signs", []float32{-1, 0, -5, 3, 2}, []int{5}, -1.0},
		{"All Negative", []float32{-10, -2, -5}, []int{3}, -17.0},
		{"Single Element", []float32{42}, []int{1}, 42.0},
		{"Empty Tensor", []float32{}, []int{0}, 0.0},
		{"Empty Tensor High Dim", []float32{}, []int{2, 0, 3}, 0.0},
		{"Empty Tensor Empty shape", []float32{}, []int{}, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(tt.data, tt.shape)
			got := tensor.Sum1()
			if math.Abs(got-tt.want) > epsilon {
				t.Errorf("Sum1() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTensor_Div1(t *testing.T) {
	t1 := NewTensor([]float32{10, 20, -30, 0}, []int{2, 2})
	t2 := NewTensor([]float32{2, 4, 5, 8}, []int{2, 2})
	result := t1.Div1(t2)
	wantData := []float32{5, 5, -6, 0}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("Div1(%v / %v) = %v, want %v", t1, t2, result, wantTensor)
	}

	t1Want := NewTensor([]float32{10, 20, -30, 0}, []int{2, 2})
	t2Want := NewTensor([]float32{2, 4, 5, 8}, []int{2, 2})
	if !tensorsEqual(t1, t1Want, epsilon) {
		t.Errorf("Original tensor t1 was modified by Div1. Got %v", t1)
	}
	if !tensorsEqual(t2, t2Want, epsilon) {
		t.Errorf("Original tensor t2 was modified by Div1. Got %v", t2)
	}

	t.Run("DivisionByZero", func(t *testing.T) {
		numerator := NewTensor([]float32{5, -5, 0, 1}, []int{4})
		denominator := NewTensor([]float32{0, 0, 0, 2}, []int{4})
		res := numerator.Div1(denominator)
		if !math.IsInf(res.Data[0], 1) {
			t.Errorf("Expected +Inf at index 0, got %v", res.Data[0])
		}
		if !math.IsInf(res.Data[1], -1) {
			t.Errorf("Expected -Inf at index 1, got %v", res.Data[1])
		}
		if !math.IsNaN(res.Data[2]) {
			t.Errorf("Expected NaN at index 2, got %v", res.Data[2])
		}
		if math.Abs(res.Data[3]-0.5) > epsilon {
			t.Errorf("Expected 0.5 at index 3, got %v", res.Data[3])
		}
	})

	t.Run("PanicShapeMismatch", func(t *testing.T) {
		t3_diff_shape := NewTensor([]float32{1, 2, 3}, []int{3})
		checkPanic(t, func() { t1.Div1(t3_diff_shape) }, "")
	})
}

func TestTensor_Multiply1(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, -3, 0}, []int{2, 2})
	t2 := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	result := t1.Multiply1(t2)
	wantData := []float32{5, 12, -21, 0}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("Multiply1(%v * %v) = %v, want %v", t1, t2, result, wantTensor)
	}

	t1Want := NewTensor([]float32{1, 2, -3, 0}, []int{2, 2})
	t2Want := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	if !tensorsEqual(t1, t1Want, epsilon) {
		t.Errorf("Original tensor t1 was modified by Multiply1. Got %v", t1)
	}
	if !tensorsEqual(t2, t2Want, epsilon) {
		t.Errorf("Original tensor t2 was modified by Multiply1. Got %v", t2)
	}

	t.Run("PanicShapeMismatch", func(t *testing.T) {
		t3_diff_shape := NewTensor([]float32{1, 2, 3}, []int{3})
		checkPanic(t, func() { t1.Multiply1(t3_diff_shape) }, "")
	})
}

func TestTensor_Apply1(t *testing.T) {
	original := NewTensor([]float32{1, 4, 9, 16}, []int{2, 2})

	t.Run("SquareRoot", func(t *testing.T) {
		result := original.Apply1(math.Sqrt)
		wantData := []float32{1, 2, 3, 4}
		wantTensor := NewTensor(wantData, []int{2, 2})
		if !tensorsEqual(result, wantTensor, epsilon) {
			t.Errorf("Apply1(sqrt) = %v, want %v", result, wantTensor)
		}
		originalWant := NewTensor([]float32{1, 4, 9, 16}, []int{2, 2})
		if !tensorsEqual(original, originalWant, epsilon) {
			t.Errorf("Original tensor was modified by Apply1(sqrt). Got %v", original)
		}
	})

	t.Run("MultiplyByTwo", func(t *testing.T) {
		multiplyByTwo := func(x float32) float32 { return x * 2.0 }
		result := original.Apply1(multiplyByTwo)
		wantData := []float32{2, 8, 18, 32}
		wantTensor := NewTensor(wantData, []int{2, 2})
		if !tensorsEqual(result, wantTensor, epsilon) {
			t.Errorf("Apply1(x*2) = %v, want %v", result, wantTensor)
		}
		originalWant := NewTensor([]float32{1, 4, 9, 16}, []int{2, 2})
		if !tensorsEqual(original, originalWant, epsilon) {
			t.Errorf("Original tensor was modified by Apply1(x*2). Got %v", original)
		}
	})

	t.Run("WithNaN", func(t *testing.T) {
		negTensor := NewTensor([]float32{-1.0, 4.0}, []int{2})
		result := negTensor.Apply1(math.Sqrt)
		if !math.IsNaN(result.Data[0]) {
			t.Errorf("Apply1(sqrt) on negative number: expected NaN, got %v", result.Data[0])
		}
		if math.Abs(result.Data[1]-2.0) > epsilon {
			t.Errorf("Apply1(sqrt) on positive number: expected 2.0, got %v", result.Data[1])
		}
	})
}

func TestTensor_Clone1(t *testing.T) {
	original := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	cloned := original.Clone1()

	if cloned == original {
		t.Errorf("Clone1 should return a new tensor instance, but got the same pointer")
	}
	if !tensorsEqual(original, cloned, epsilon) {
		t.Errorf("Cloned tensor is not equal to original. Got %v, want %v", cloned, original)
	}
	if len(original.Data) > 0 && &original.Data[0] == &cloned.Data[0] {
		t.Errorf("Clone1 did not create a deep copy of Data slice")
	}
	if original.shape != nil && &original.shape == &cloned.shape {
		t.Errorf("Clone1 did not create a deep copy of shape slice")
	}

	original.Data[0] = 99.0
	if original.shape != nil && len(original.shape) > 0 {
		original.shape[0] = 55
	}

	if cloned.Data[0] == 99.0 {
		t.Errorf("Modifying original tensor affected the cloned tensor's data")
	}
	if cloned.shape != nil && len(cloned.shape) > 0 && cloned.shape[0] == 55 {
		t.Errorf("Modifying original tensor's shape slice affected the cloned tensor's shape slice")
	}

	wantShape := []int{2, 2}
	if !reflect.DeepEqual(cloned.shape, wantShape) {
		t.Errorf("Cloned tensor shape changed unexpectedly. Got %v, want %v", cloned.shape, wantShape)
	}

	empty := NewTensor([]float32{}, []int{0, 2})
	clonedEmpty := empty.Clone1()
	wantEmpty := NewTensor([]float32{}, []int{0, 2})
	if !tensorsEqual(clonedEmpty, wantEmpty, epsilon) || clonedEmpty == empty {
		t.Errorf("Cloning empty tensor failed. Got %v, want %v (different instance)", clonedEmpty, wantEmpty)
	}

	scalar := NewTensor([]float32{5.0}, []int{1})
	clonedScalar := scalar.Clone1()
	if !tensorsEqual(scalar, clonedScalar, epsilon) || scalar == clonedScalar {
		t.Errorf("Cloning scalar tensor failed. Got %v, want %v (different instance)", clonedScalar, scalar)
	}
	scalar.Data[0] = 6.0
	if clonedScalar.Data[0] == 6.0 {
		t.Errorf("Modifying original scalar tensor affected clone")
	}
}

func TestShapeEqual(t *testing.T) {
	tests := []struct {
		name   string
		shape1 []int
		shape2 []int
		want   bool
	}{
		{"Identical Shapes", []int{2, 3}, []int{2, 3}, true},
		{"Different Dimensions", []int{2, 3}, []int{2, 4}, false},
		{"Different Ranks", []int{2, 3}, []int{2, 3, 1}, false},
		{"One Nil, One Empty", nil, []int{}, false},
		{"Both Nil", nil, nil, true},
		{"Both Empty", []int{}, []int{}, true},
		{"One Nil, One Valid", nil, []int{2, 3}, false},
		{"One Empty, One Valid", []int{}, []int{2, 3}, false},
		{"Scalar vs Scalar", []int{1}, []int{1}, true},
		{"Scalar vs Empty", []int{1}, []int{}, false},
		{"Scalar vs Nil", []int{1}, nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shapeEqual(tt.shape1, tt.shape2); got != tt.want {
				t.Errorf("unexported shapeEqual(%v, %v) = %v, want %v", tt.shape1, tt.shape2, got, tt.want)
			}
			if got := ShapeEqual(tt.shape1, tt.shape2); got != tt.want {
				t.Errorf("exported ShapeEqual(%v, %v) = %v, want %v", tt.shape1, tt.shape2, got, tt.want)
			}
		})
	}
}

func TestTensor_SubScalar(t *testing.T) {
	original := NewTensor([]float32{10, 20, 30, 0}, []int{2, 2})
	scalar := float32(5.0)
	result := original.SubScalar(scalar)
	wantData := []float32{5, 15, 25, -5}
	wantTensor := NewTensor(wantData, []int{2, 2})

	if !tensorsEqual(result, wantTensor, epsilon) {
		t.Errorf("SubScalar(%v) = %v, want %v", scalar, result, wantTensor)
	}

	originalWant := NewTensor([]float32{10, 20, 30, 0}, []int{2, 2})
	if !tensorsEqual(original, originalWant, epsilon) {
		t.Errorf("Original tensor was modified by SubScalar. Got %v, want %v", original, originalWant)
	}

	t.Run("SubtractZero", func(t *testing.T) {
		resZero := original.SubScalar(0.0)
		if !tensorsEqual(resZero, originalWant, epsilon) {
			t.Errorf("SubScalar(0.0) = %v, want %v", resZero, originalWant)
		}
	})

	t.Run("SubtractNegative", func(t *testing.T) {
		resNeg := original.SubScalar(-2.0)
		wantNegData := []float32{12, 22, 32, 2}
		wantNegTensor := NewTensor(wantNegData, []int{2, 2})
		if !tensorsEqual(resNeg, wantNegTensor, epsilon) {
			t.Errorf("SubScalar(-2.0) = %v, want %v", resNeg, wantNegTensor)
		}
	})
}
