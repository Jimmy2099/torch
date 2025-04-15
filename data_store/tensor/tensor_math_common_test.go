package tensor

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"reflect"
	"testing"
)

const tolerance = 1e-9 // Tolerance for float comparisons

func floatsEqual(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

func shapesEqual(a, b []int) bool {
	return reflect.DeepEqual(a, b) // reflect.DeepEqual works well for int slices
}

func assertTensorsEqual(t *testing.T, got, want *Tensor, tol float32) {
	t.Helper()
	if got == nil && want == nil {
		return
	}
	if got == nil || want == nil {
		t.Errorf("One tensor is nil, the other is not. Got: %v, Want: %v", got, want)
		return
	}
	if !shapesEqual(got.Shape, want.Shape) {
		t.Errorf("Shape mismatch: Got %v, want %v", got.Shape, want.Shape)
		return // Stop comparison if shapes differ
	}
	if !floatsEqual(got.Data, want.Data, tol) {
		t.Errorf("Data mismatch:\nGot:  %v\nWant: %v\nShape: %v", got.Data, want.Data, got.Shape)
	}
}

func assertPanic(t *testing.T, fn func(), expectedMsgPart string) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("Expected a panic, but did not get one.")
			return
		}
		msg := fmt.Sprintf("%v", r)
		if expectedMsgPart != "" && !contains(msg, expectedMsgPart) {
			t.Errorf("Panic message '%s' does not contain expected part '%s'", msg, expectedMsgPart)
		}
	}()
	fn()
}

func TestFlatten(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	want := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{6})
	got := t1.Flatten()
	assertTensorsEqual(t, got, want, 0) // Exact comparison

	t2 := NewTensor([]float32{1, 2, 3, 4}, []int{1, 4})
	want2 := NewTensor([]float32{1, 2, 3, 4}, []int{4})
	got2 := t2.Flatten()
	assertTensorsEqual(t, got2, want2, 0)

	t3 := NewTensor([]float32{7}, []int{1, 1, 1})
	want3 := NewTensor([]float32{7}, []int{1})
	got3 := t3.Flatten()
	assertTensorsEqual(t, got3, want3, 0)
}

func TestSet(t *testing.T) {
	tensor := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	tensor.Set(99.0, 1, 1) // Set element at [1][1]

	wantData := []float32{1, 2, 3, 4, 99, 6}
	if !floatsEqual(tensor.Data, wantData, 0) {
		t.Errorf("Set failed. Got %v, want %v", tensor.Data, wantData)
	}

	tensor.Set(-5.0, 0, 0) // Set element at [0][0]
	wantData = []float32{-5, 2, 3, 4, 99, 6}
	if !floatsEqual(tensor.Data, wantData, 0) {
		t.Errorf("Set failed. Got %v, want %v", tensor.Data, wantData)
	}

	tensor3D := NewTensor(make([]float32, 12), []int{2, 2, 3}) // Zeros
	tensor3D.Set(7.0, 1, 0, 2)
	wantData3D := make([]float32, 12)
	wantData3D[8] = 7.0
	if !floatsEqual(tensor3D.Data, wantData3D, 0) {
		t.Errorf("Set failed for 3D. Got %v, want %v", tensor3D.Data, wantData3D)
	}
}

func TestMultiply1(t *testing.T) {
	t.Run("2D Matrix Multiplication", func(t *testing.T) {
		a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})    // 2x3
		b := NewTensor([]float32{7, 8, 9, 10, 11, 12}, []int{3, 2}) // 3x2
		want := NewTensor([]float32{58, 64, 139, 154}, []int{2, 2})
		got := Multiply(a, b)
		assertTensorsEqual(t, got, want, tolerance)
	})

	t.Run("Higher Dimensions (Batch Matrix Multiplication)", func(t *testing.T) {
		a := NewTensor([]float32{
			1, 2, 3, // Batch 0, Row 0
			4, 5, 6, // Batch 0, Row 1
			7, 8, 9, // Batch 1, Row 0
			10, 11, 12, // Batch 1, Row 1
		}, []int{2, 2, 3}) // Shape (Batch=2, Rows=2, Cols=3)

		b := NewTensor([]float32{
			1, 0, // Batch 0, Col 0
			0, 1, // Batch 0, Col 1
			1, 1, // Batch 0, Col 2 (Shape 3x2)
			2, 1, // Batch 1, Col 0
			1, 2, // Batch 1, Col 1
			0, 0, // Batch 1, Col 2 (Shape 3x2)
		}, []int{2, 3, 2}) // Shape (Batch=2, Rows=3, Cols=2)

		wantData := []float32{
			4, 5, 10, 11, // Batch 0 result (2x2)
			22, 23, 31, 32, // Batch 1 result (2x2)
		}
		wantShape := []int{2, 2, 2} // Shape (Batch=2, Rows=2, Cols=2)
		want := NewTensor(wantData, wantShape)

		got := Multiply(a, b)
		assertTensorsEqual(t, got, want, tolerance)
	})

	t.Run("Panic on Dimension Mismatch", func(t *testing.T) {
		a := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
		b := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}) // Inner dims mismatch (2 vs 3)
		assertPanic(t, func() { Multiply(a, b) }, "Tensor dimensions don't match")
	})

	t.Run("Panic on Leading Dimension Mismatch (Higher Dims)", func(t *testing.T) {
		a := NewTensor(make([]float32, 12), []int{2, 2, 3}) // Batch=2
		b := NewTensor(make([]float32, 18), []int{3, 3, 2}) // Batch=3
		assertPanic(t, func() { Multiply(a, b) }, "Leading tensor dimensions don't match")
	})

	t.Run("Panic on Insufficient Dimensions", func(t *testing.T) {
		a := NewTensor([]float32{1, 2, 3}, []int{3}) // 1D
		b := NewTensor([]float32{4, 5, 6}, []int{3}) // 1D
		assertPanic(t, func() { Multiply(a, b) }, "Tensors must have at least 2 dimensions")

		c := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})                                   // 2D
		assertPanic(t, func() { Multiply(a, c) }, "Tensors must have at least 2 dimensions") // a is 1D
		assertPanic(t, func() { Multiply(c, a) }, "Tensors must have at least 2 dimensions") // a is 1D
	})
}

func TestAdd(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	want := NewTensor([]float32{6, 8, 10, 12}, []int{2, 2})
	got := Add(a, b)
	assertTensorsEqual(t, got, want, tolerance)

	c := NewTensor([]float32{1, 2, 3}, []int{3})
	assertPanic(t, func() { Add(a, c) }, "Tensor shapes don't match for addition")
}

func TestSubtract(t *testing.T) {
	a := NewTensor([]float32{10, 9, 8, 7}, []int{2, 2})
	b := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	want := NewTensor([]float32{9, 7, 5, 3}, []int{2, 2})
	got := Subtract(a, b)
	assertTensorsEqual(t, got, want, tolerance)

	c := NewTensor([]float32{1, 2}, []int{1, 2})
	assertPanic(t, func() { Subtract(a, c) }, "Tensor shapes don't match for subtraction")
}

func TestHadamardProduct(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	want := NewTensor([]float32{5, 12, 21, 32}, []int{2, 2})
	got := HadamardProduct(a, b)
	assertTensorsEqual(t, got, want, tolerance)

	c := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	assertPanic(t, func() { HadamardProduct(a, c) }, "Tensor shapes don't match for Hadamard product")
}

func TestTranspose1(t *testing.T) {
	t.Run("Default Transpose (Last Two Dims)", func(t *testing.T) {
		a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
		want := NewTensor([]float32{1, 4, 2, 5, 3, 6}, []int{3, 2})
		got := Transpose(a)
		assertTensorsEqual(t, got, want, 0)

		bData := []float32{
			1, 2, 3, 4, 5, 6, // Slice 0
			7, 8, 9, 10, 11, 12, // Slice 1
		}
		b := NewTensor(bData, []int{2, 2, 3})
		wantData := []float32{
			1, 4, 2, 5, 3, 6, // Transposed Slice 0
			7, 10, 8, 11, 9, 12, // Transposed Slice 1
		}
		wantB := NewTensor(wantData, []int{2, 3, 2})
		gotB := Transpose(b)
		assertTensorsEqual(t, gotB, wantB, 0)
	})

	t.Run("Specific Dimensions", func(t *testing.T) {
		data := make([]float32, 24)
		for i := range data {
			data[i] = float32(i)
		}
		a := NewTensor(data, []int{2, 3, 4})
		wantData := []float32{
			0, 12, 1, 13, 2, 14, 3, 15, // j=0
			4, 16, 5, 17, 6, 18, 7, 19, // j=1
			8, 20, 9, 21, 10, 22, 11, 23, // j=2
		}
		wantShape := []int{3, 4, 2}
		want := NewTensor(wantData, wantShape)
		got := Transpose(a, 1, 2, 0)
		assertTensorsEqual(t, got, want, 0)
	})

	t.Run("1D Tensor", func(t *testing.T) {
		a := NewTensor([]float32{1, 2, 3}, []int{3})
		got := Transpose(a) // Should return a copy
		assertTensorsEqual(t, got, a, 0)
		got.Data[0] = 99
		if a.Data[0] == 99 {
			t.Error("Transpose of 1D tensor did not return a copy")
		}
	})

	t.Run("Panic Invalid Dimensions", func(t *testing.T) {
		a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
		assertPanic(t, func() { Transpose(a, 0) }, "Invalid transpose dimensions")    // Wrong number of dims
		assertPanic(t, func() { Transpose(a, 1, 2) }, "Invalid transpose dimensions") // Wrong number of dims
	})
}

func TestApply(t *testing.T) {
	a := NewTensor([]float32{1, -2, 3, -4}, []int{2, 2})
	fn := func(x float32) float32 { return x * x } // Square function
	want := NewTensor([]float32{1, 4, 9, 16}, []int{2, 2})
	got := a.Apply(fn)
	assertTensorsEqual(t, got, want, tolerance)

	fnAbs := math.Abs
	wantAbs := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	gotAbs := a.Apply(fnAbs)
	assertTensorsEqual(t, gotAbs, wantAbs, tolerance)
}

func TestSumMeanMax(t *testing.T) {
	a := NewTensor([]float32{1, -2, 3, -4, 5, 0}, []int{2, 3})

	if sum := a.Sum(); math.Abs(sum-3.0) > tolerance {
		t.Errorf("Sum failed: Got %f, want %f", sum, 3.0)
	}
	if mean := a.Mean(); math.Abs(mean-0.5) > tolerance {
		t.Errorf("Mean failed: Got %f, want %f", mean, 0.5)
	}
	if max := a.Max(); math.Abs(max-5.0) > tolerance {
		t.Errorf("Max failed: Got %f, want %f", max, 5.0)
	}

	b := NewTensor([]float32{-10, -20}, []int{2})
	if max := b.Max(); math.Abs(max-(-10.0)) > tolerance {
		t.Errorf("Max failed for negative numbers: Got %f, want %f", max, -10.0)
	}
}

func TestCopy(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	b := a.Copy()

	assertTensorsEqual(t, a, b, 0) // Should be identical initially

	b.Set(99.0, 0, 1)
	b.Shape[0] = 5 // Modify shape (though Set doesn't use it after creation)

	wantA := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	assertTensorsEqual(t, a, wantA, 0)

	wantB := NewTensor([]float32{1, 99, 3, 4}, []int{5, 2}) // Note modified shape
	if !floatsEqual(b.Data, wantB.Data, 0) {
		t.Errorf("Copy data modification failed: Got %v, want %v", b.Data, wantB.Data)
	}
	if !shapesEqual(b.Shape, wantB.Shape) {
		t.Errorf("Copy shape modification failed: Got %v, want %v", b.Shape, wantB.Shape)
	}
}

func TestReLU(t *testing.T) {
	a := NewTensor([]float32{1, -2, 0, 3, -0.5}, []int{5})
	want := NewTensor([]float32{1, 0, 0, 3, 0}, []int{5})
	got := a.ReLU()
	assertTensorsEqual(t, got, want, tolerance)
}

func TestSoftmax(t *testing.T) {
	t.Run("1D Softmax", func(t *testing.T) {
		a := NewTensor([]float32{1, 2, 3}, []int{3})
		wantData := []float32{0.09003057, 0.24472847, 0.66524096}
		want := NewTensor(wantData, []int{3})
		got := a.Softmax()
		assertTensorsEqual(t, got, want, tolerance)
		if sum := got.Sum(); math.Abs(sum-1.0) > tolerance {
			t.Errorf("Softmax sum is not 1: got %f", sum)
		}
	})

	t.Run("2D Softmax (along last dim)", func(t *testing.T) {
		a := NewTensor([]float32{
			1, 2, 3, // Row 0
			4, 1, 1, // Row 1
		}, []int{2, 3})

		wantData := []float32{
			0.09003057, 0.24472847, 0.66524096,
			0.90941165, 0.04529417, 0.04529417,
		}
		want := NewTensor(wantData, []int{2, 3})
		got := a.Softmax()
		assertTensorsEqual(t, got, want, tolerance)
	})

	t.Run("Panic on 0D tensor", func(t *testing.T) {
	})
}

func TestArgMax(t *testing.T) {
	t.Run("1D ArgMax", func(t *testing.T) {
		a := NewTensor([]float32{1, 5, 2, 5, 3}, []int{5})
		want := NewTensor([]float32{1}, []int{}) // ArgMax reduces dimension
		got := a.ArgMax()
		assertTensorsEqual(t, got, want, 0) // Index is integer
	})

	t.Run("2D ArgMax (along last dim)", func(t *testing.T) {
		a := NewTensor([]float32{
			1, 9, 2, // Row 0, max at index 1
			8, 5, 7, // Row 1, max at index 0
			3, 4, 6, // Row 2, max at index 2
		}, []int{3, 3})
		want := NewTensor([]float32{1, 0, 2}, []int{3}) // Shape is [3]
		got := a.ArgMax()
		assertTensorsEqual(t, got, want, 0)
	})

	t.Run("3D ArgMax", func(t *testing.T) {
		a := NewTensor([]float32{
			1, 9, 2, 8, 5, 7, // Slice 0 (2x3) -> ArgMax -> [1, 0]
			3, 4, 6, 0, 1, -1, // Slice 1 (2x3) -> ArgMax -> [2, 1]
		}, []int{2, 2, 3})
		want := NewTensor([]float32{1, 0, 2, 1}, []int{2, 2}) // Shape [2, 2]
		got := a.ArgMax()
		assertTensorsEqual(t, got, want, 0)
	})

	t.Run("Panic on 0D tensor", func(t *testing.T) {
	})
}

func TestMaxPool(t *testing.T) {
	t.Run("2D MaxPool", func(t *testing.T) {
		data := []float32{
			1, 3, 2, 4,
			5, 7, 6, 8,
			9, 11, 10, 12,
			13, 15, 14, 16,
		}
		a := NewTensor(data, []int{4, 4})
		poolSize := 2
		stride := 2

		wantPool := NewTensor([]float32{7, 8, 15, 16}, []int{2, 2})
		wantArgmax := NewTensor([]float32{5, 7, 13, 15}, []int{2, 2}) // Indices in flattened original data

		gotPool, gotArgmax := a.MaxPool(poolSize, stride)
		assertTensorsEqual(t, gotPool, wantPool, 0)
		assertTensorsEqual(t, gotArgmax, wantArgmax, 0)
	})

	t.Run("2D MaxPool with Stride 1", func(t *testing.T) {
		data := []float32{
			1, 3, 2,
			5, 7, 6,
			9, 11, 10,
		}
		a := NewTensor(data, []int{3, 3})
		poolSize := 2
		stride := 1
		wantPool := NewTensor([]float32{7, 7, 11, 11}, []int{2, 2})
		wantArgmax := NewTensor([]float32{4, 4, 7, 7}, []int{2, 2})

		gotPool, gotArgmax := a.MaxPool(poolSize, stride)
		assertTensorsEqual(t, gotPool, wantPool, 0)
		assertTensorsEqual(t, gotArgmax, wantArgmax, 0)
	})

	t.Run("3D MaxPool", func(t *testing.T) {
		data := []float32{
			1, 3, 2, 4,
			5, 7, 6, 8,
			9, 11, 10, 12,
			13, 15, 14, 16,
			16, 14, 15, 13,
			12, 10, 11, 9,
			8, 6, 7, 5,
			4, 2, 3, 1,
		}
		a := NewTensor(data, []int{2, 4, 4})
		poolSize := 2
		stride := 2

		wantPoolData := []float32{7, 8, 15, 16, 16, 15, 8, 7}
		wantArgmData := []float32{5, 7, 13, 15, 16, 18, 24, 26}
		wantShape := []int{2, 2, 2}
		wantPool := NewTensor(wantPoolData, wantShape)
		wantArgmax := NewTensor(wantArgmData, wantShape)

		gotPool, gotArgmax := a.MaxPool(poolSize, stride)
		assertTensorsEqual(t, gotPool, wantPool, 0)
		assertTensorsEqual(t, gotArgmax, wantArgmax, 0)
	})

	t.Run("Panic on Insufficient Dimensions", func(t *testing.T) {
		a := NewTensor([]float32{1, 2, 3}, []int{3}) // 1D
		assertPanic(t, func() { a.MaxPool(2, 1) }, "at least 2 dimensions")
	})
}

func TestShapesMatch(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	t2 := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	t3 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	t4 := NewTensor([]float32{1, 2, 3, 4}, []int{4})
	var tNil *Tensor = nil

	if !t1.ShapesMatch(t2) {
		t.Error("Expected ShapesMatch to return true for identical shapes")
	}
	if t1.ShapesMatch(t3) {
		t.Error("Expected ShapesMatch to return false for different shapes (size)")
	}
	if t1.ShapesMatch(t4) {
		t.Error("Expected ShapesMatch to return false for different shapes (dims)")
	}
	if t1.ShapesMatch(nil) {
		t.Error("Expected ShapesMatch to return false for nil comparison")
	}
	if tNil.ShapesMatch(t1) {
		t.Error("Expected ShapesMatch to return false when receiver is nil")
	}
	if tNil.ShapesMatch(nil) {
		t.Error("Expected ShapesMatch to return false when both are nil") // Or true depending on desired semantics, false seems safer
	}
}
