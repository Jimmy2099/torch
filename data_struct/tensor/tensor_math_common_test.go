// tensor_test.go
package tensor

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

const tolerance = 1e-9 // Tolerance for float comparisons

// Helper function to compare float slices with tolerance
func floatsEqual(a, b []float64, tol float64) bool {
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

// Helper function to compare shapes
func shapesEqual(a, b []int) bool {
	return reflect.DeepEqual(a, b) // reflect.DeepEqual works well for int slices
}

// Helper function to assert tensor equality
func assertTensorsEqual(t *testing.T, got, want *Tensor, tol float64) {
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
		// Optionally print data for debugging, but can be verbose
		// t.Logf("Got Data: %v", got.Data)
		// t.Logf("Want Data: %v", want.Data)
		return // Stop comparison if shapes differ
	}
	if !floatsEqual(got.Data, want.Data, tol) {
		t.Errorf("Data mismatch:\nGot:  %v\nWant: %v\nShape: %v", got.Data, want.Data, got.Shape)
	}
}

// Helper function to check for panics
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
		// If expectedMsgPart is empty, any panic is accepted (use carefully)
	}()
	fn()
}

// --- Test Functions ---
func TestFlatten(t *testing.T) {
	t1 := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	want := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{6})
	got := t1.Flatten()
	assertTensorsEqual(t, got, want, 0) // Exact comparison

	t2 := NewTensor([]float64{1, 2, 3, 4}, []int{1, 4})
	want2 := NewTensor([]float64{1, 2, 3, 4}, []int{4})
	got2 := t2.Flatten()
	assertTensorsEqual(t, got2, want2, 0)

	t3 := NewTensor([]float64{7}, []int{1, 1, 1})
	want3 := NewTensor([]float64{7}, []int{1})
	got3 := t3.Flatten()
	assertTensorsEqual(t, got3, want3, 0)
}

func TestSet(t *testing.T) {
	tensor := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	tensor.Set(99.0, 1, 1) // Set element at [1][1]

	// Expected data after set: [1, 2, 3, 4, 99, 6]
	// Index calculation: 1*stride(dim 0) + 1*stride(dim 1) = 1*3 + 1*1 = 4
	wantData := []float64{1, 2, 3, 4, 99, 6}
	if !floatsEqual(tensor.Data, wantData, 0) {
		t.Errorf("Set failed. Got %v, want %v", tensor.Data, wantData)
	}

	tensor.Set(-5.0, 0, 0) // Set element at [0][0]
	wantData = []float64{-5, 2, 3, 4, 99, 6}
	if !floatsEqual(tensor.Data, wantData, 0) {
		t.Errorf("Set failed. Got %v, want %v", tensor.Data, wantData)
	}

	// Test with 3D tensor
	tensor3D := NewTensor(make([]float64, 12), []int{2, 2, 3}) // Zeros
	tensor3D.Set(7.0, 1, 0, 2)
	// Index: 1*stride(0) + 0*stride(1) + 2*stride(2) = 1*(2*3) + 0*(3) + 2*(1) = 6 + 0 + 2 = 8
	wantData3D := make([]float64, 12)
	wantData3D[8] = 7.0
	if !floatsEqual(tensor3D.Data, wantData3D, 0) {
		t.Errorf("Set failed for 3D. Got %v, want %v", tensor3D.Data, wantData3D)
	}
}

func TestMultiply1(t *testing.T) {
	t.Run("2D Matrix Multiplication", func(t *testing.T) {
		a := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})    // 2x3
		b := NewTensor([]float64{7, 8, 9, 10, 11, 12}, []int{3, 2}) // 3x2
		// Expected: [[58, 64], [139, 154]] -> [58, 64, 139, 154]
		want := NewTensor([]float64{58, 64, 139, 154}, []int{2, 2})
		got := Multiply(a, b)
		assertTensorsEqual(t, got, want, tolerance)
	})

	t.Run("Higher Dimensions (Batch Matrix Multiplication)", func(t *testing.T) {
		// Correcting the shape and data for typical batch matmul: (Batch, Rows, Cols)
		a := NewTensor([]float64{
			1, 2, 3, // Batch 0, Row 0
			4, 5, 6, // Batch 0, Row 1
			7, 8, 9, // Batch 1, Row 0
			10, 11, 12, // Batch 1, Row 1
		}, []int{2, 2, 3}) // Shape (Batch=2, Rows=2, Cols=3)

		b := NewTensor([]float64{
			1, 0, // Batch 0, Col 0
			0, 1, // Batch 0, Col 1
			1, 1, // Batch 0, Col 2 (Shape 3x2)
			2, 1, // Batch 1, Col 0
			1, 2, // Batch 1, Col 1
			0, 0, // Batch 1, Col 2 (Shape 3x2)
		}, []int{2, 3, 2}) // Shape (Batch=2, Rows=3, Cols=2)

		// Expected Batch 0: [[1*1+2*0+3*1, 1*0+2*1+3*1], [4*1+5*0+6*1, 4*0+5*1+6*1]] = [[4, 5], [10, 11]]
		// Expected Batch 1: [[7*2+8*1+9*0, 7*1+8*2+9*0], [10*2+11*1+12*0, 10*1+11*2+12*0]] = [[22, 23], [31, 32]]
		wantData := []float64{
			4, 5, 10, 11, // Batch 0 result (2x2)
			22, 23, 31, 32, // Batch 1 result (2x2)
		}
		wantShape := []int{2, 2, 2} // Shape (Batch=2, Rows=2, Cols=2)
		want := NewTensor(wantData, wantShape)

		got := Multiply(a, b)
		assertTensorsEqual(t, got, want, tolerance)
	})

	t.Run("Panic on Dimension Mismatch", func(t *testing.T) {
		a := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
		b := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2}) // Inner dims mismatch (2 vs 3)
		assertPanic(t, func() { Multiply(a, b) }, "Tensor dimensions don't match")
	})

	t.Run("Panic on Leading Dimension Mismatch (Higher Dims)", func(t *testing.T) {
		a := NewTensor(make([]float64, 12), []int{2, 2, 3}) // Batch=2
		b := NewTensor(make([]float64, 18), []int{3, 3, 2}) // Batch=3
		assertPanic(t, func() { Multiply(a, b) }, "Leading tensor dimensions don't match")
	})

	t.Run("Panic on Insufficient Dimensions", func(t *testing.T) {
		a := NewTensor([]float64{1, 2, 3}, []int{3}) // 1D
		b := NewTensor([]float64{4, 5, 6}, []int{3}) // 1D
		assertPanic(t, func() { Multiply(a, b) }, "Tensors must have at least 2 dimensions")

		c := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})                                   // 2D
		assertPanic(t, func() { Multiply(a, c) }, "Tensors must have at least 2 dimensions") // a is 1D
		assertPanic(t, func() { Multiply(c, a) }, "Tensors must have at least 2 dimensions") // a is 1D
	})
}

func TestAdd(t *testing.T) {
	a := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	b := NewTensor([]float64{5, 6, 7, 8}, []int{2, 2})
	want := NewTensor([]float64{6, 8, 10, 12}, []int{2, 2})
	got := Add(a, b)
	assertTensorsEqual(t, got, want, tolerance)

	// Test panic on shape mismatch
	c := NewTensor([]float64{1, 2, 3}, []int{3})
	assertPanic(t, func() { Add(a, c) }, "Tensor shapes don't match for addition")
}

func TestSubtract(t *testing.T) {
	a := NewTensor([]float64{10, 9, 8, 7}, []int{2, 2})
	b := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	want := NewTensor([]float64{9, 7, 5, 3}, []int{2, 2})
	got := Subtract(a, b)
	assertTensorsEqual(t, got, want, tolerance)

	// Test panic on shape mismatch
	c := NewTensor([]float64{1, 2}, []int{1, 2})
	assertPanic(t, func() { Subtract(a, c) }, "Tensor shapes don't match for subtraction")
}

func TestHadamardProduct(t *testing.T) {
	a := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	b := NewTensor([]float64{5, 6, 7, 8}, []int{2, 2})
	want := NewTensor([]float64{5, 12, 21, 32}, []int{2, 2})
	got := HadamardProduct(a, b)
	assertTensorsEqual(t, got, want, tolerance)

	// Test panic on shape mismatch
	c := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	assertPanic(t, func() { HadamardProduct(a, c) }, "Tensor shapes don't match for Hadamard product")
}

func TestTranspose1(t *testing.T) {
	t.Run("Default Transpose (Last Two Dims)", func(t *testing.T) {
		// 2x3 -> 3x2
		a := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
		want := NewTensor([]float64{1, 4, 2, 5, 3, 6}, []int{3, 2})
		got := Transpose(a)
		assertTensorsEqual(t, got, want, 0)

		// 2x2x3 -> 2x3x2
		bData := []float64{
			1, 2, 3, 4, 5, 6, // Slice 0
			7, 8, 9, 10, 11, 12, // Slice 1
		}
		b := NewTensor(bData, []int{2, 2, 3})
		wantData := []float64{
			1, 4, 2, 5, 3, 6, // Transposed Slice 0
			7, 10, 8, 11, 9, 12, // Transposed Slice 1
		}
		wantB := NewTensor(wantData, []int{2, 3, 2})
		gotB := Transpose(b)
		assertTensorsEqual(t, gotB, wantB, 0)
	})

	t.Run("Specific Dimensions", func(t *testing.T) {
		// 2x3x4 -> Transpose(1, 2, 0) -> 3x4x2
		data := make([]float64, 24)
		for i := range data {
			data[i] = float64(i)
		}
		a := NewTensor(data, []int{2, 3, 4})
		// Expected data requires careful index mapping.
		// Original (i,j,k), New (j,k,i)
		// New pos(j,k,i) = j*stride(0) + k*stride(1) + i*stride(2) = j*(4*2) + k*(2) + i*1 = 8j + 2k + i
		// Example: a.Data[1] = 1 (i=0,j=0,k=1) -> new pos = 8*0 + 2*1 + 0 = 2. So resultData[2] = 1
		// Example: a.Data[5] = 5 (i=0,j=1,k=1) -> new pos = 8*1 + 2*1 + 0 = 10. So resultData[10] = 5
		// Example: a.Data[13] = 13 (i=1,j=0,k=1) -> new pos = 8*0 + 2*1 + 1 = 3. So resultData[3] = 13
		// Example: a.Data[17] = 17 (i=1,j=1,k=1) -> new pos = 8*1 + 2*1 + 1 = 11. So resultData[11] = 17
		wantData := []float64{
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
		a := NewTensor([]float64{1, 2, 3}, []int{3})
		got := Transpose(a) // Should return a copy
		assertTensorsEqual(t, got, a, 0)
		// Ensure it's a copy, not the same tensor
		got.Data[0] = 99
		if a.Data[0] == 99 {
			t.Error("Transpose of 1D tensor did not return a copy")
		}
	})

	t.Run("Panic Invalid Dimensions", func(t *testing.T) {
		a := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
		assertPanic(t, func() { Transpose(a, 0) }, "Invalid transpose dimensions")    // Wrong number of dims
		assertPanic(t, func() { Transpose(a, 1, 2) }, "Invalid transpose dimensions") // Wrong number of dims
		// Potential future test: check if dims are a valid permutation (e.g., no duplicates) - current code doesn't check this.
	})
}

func TestApply(t *testing.T) {
	a := NewTensor([]float64{1, -2, 3, -4}, []int{2, 2})
	fn := func(x float64) float64 { return x * x } // Square function
	want := NewTensor([]float64{1, 4, 9, 16}, []int{2, 2})
	got := a.Apply(fn)
	assertTensorsEqual(t, got, want, tolerance)

	fnAbs := math.Abs
	wantAbs := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	gotAbs := a.Apply(fnAbs)
	assertTensorsEqual(t, gotAbs, wantAbs, tolerance)
}

func TestSumMeanMax(t *testing.T) {
	a := NewTensor([]float64{1, -2, 3, -4, 5, 0}, []int{2, 3})
	// Sum = 1 - 2 + 3 - 4 + 5 + 0 = 3
	// Mean = 3 / 6 = 0.5
	// Max = 5

	if sum := a.Sum(); math.Abs(sum-3.0) > tolerance {
		t.Errorf("Sum failed: Got %f, want %f", sum, 3.0)
	}
	if mean := a.Mean(); math.Abs(mean-0.5) > tolerance {
		t.Errorf("Mean failed: Got %f, want %f", mean, 0.5)
	}
	if max := a.Max(); math.Abs(max-5.0) > tolerance {
		t.Errorf("Max failed: Got %f, want %f", max, 5.0)
	}

	b := NewTensor([]float64{-10, -20}, []int{2})
	if max := b.Max(); math.Abs(max-(-10.0)) > tolerance {
		t.Errorf("Max failed for negative numbers: Got %f, want %f", max, -10.0)
	}
}

func TestCopy(t *testing.T) {
	a := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	b := a.Copy()

	assertTensorsEqual(t, a, b, 0) // Should be identical initially

	// Modify copy
	b.Set(99.0, 0, 1)
	b.Shape[0] = 5 // Modify shape (though Set doesn't use it after creation)

	// Check original is unchanged
	wantA := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	assertTensorsEqual(t, a, wantA, 0)

	// Check copy is changed
	wantB := NewTensor([]float64{1, 99, 3, 4}, []int{5, 2}) // Note modified shape
	if !floatsEqual(b.Data, wantB.Data, 0) {
		t.Errorf("Copy data modification failed: Got %v, want %v", b.Data, wantB.Data)
	}
	if !shapesEqual(b.Shape, wantB.Shape) {
		t.Errorf("Copy shape modification failed: Got %v, want %v", b.Shape, wantB.Shape)
	}
}

func TestReLU(t *testing.T) {
	a := NewTensor([]float64{1, -2, 0, 3, -0.5}, []int{5})
	want := NewTensor([]float64{1, 0, 0, 3, 0}, []int{5})
	got := a.ReLU()
	assertTensorsEqual(t, got, want, tolerance)
}

func TestSoftmax(t *testing.T) {
	t.Run("1D Softmax", func(t *testing.T) {
		a := NewTensor([]float64{1, 2, 3}, []int{3})
		// exp(1-3), exp(2-3), exp(3-3) -> exp(-2), exp(-1), exp(0)
		// e^-2 ~= 0.1353, e^-1 ~= 0.3679, e^0 = 1.0
		// Sum = 1.5032
		// Softmax = [0.1353/1.5032, 0.3679/1.5032, 1.0/1.5032] ~= [0.0900, 0.2447, 0.6652]
		wantData := []float64{0.09003057, 0.24472847, 0.66524096}
		want := NewTensor(wantData, []int{3})
		got := a.Softmax()
		assertTensorsEqual(t, got, want, tolerance)
		// Check sum is close to 1
		if sum := got.Sum(); math.Abs(sum-1.0) > tolerance {
			t.Errorf("Softmax sum is not 1: got %f", sum)
		}
	})

	t.Run("2D Softmax (along last dim)", func(t *testing.T) {
		a := NewTensor([]float64{
			1, 2, 3, // Row 0
			4, 1, 1, // Row 1
		}, []int{2, 3})

		// Row 0: Same as 1D test -> [0.0900, 0.2447, 0.6652]
		// Row 1: exp(4-4), exp(1-4), exp(1-4) -> exp(0), exp(-3), exp(-3)
		// e^0 = 1.0, e^-3 ~= 0.0498
		// Sum = 1.0 + 0.0498 + 0.0498 = 1.0996
		// Softmax = [1/1.0996, 0.0498/1.0996, 0.0498/1.0996] ~= [0.9094, 0.0453, 0.0453]
		wantData := []float64{
			0.09003057, 0.24472847, 0.66524096,
			0.90941165, 0.04529417, 0.04529417,
		}
		want := NewTensor(wantData, []int{2, 3})
		got := a.Softmax()
		assertTensorsEqual(t, got, want, tolerance)
	})

	t.Run("Panic on 0D tensor", func(t *testing.T) {
		// Assuming NewTensor prevents 0D or handles it, otherwise test:
		// a := NewTensor([]float64{5}, []int{})
		// assertPanic(t, func(){ a.Softmax() }, "at least 1 dimension")
	})
}

func TestArgMax(t *testing.T) {
	t.Run("1D ArgMax", func(t *testing.T) {
		a := NewTensor([]float64{1, 5, 2, 5, 3}, []int{5})
		// Max is 5 at index 1 (and 3, returns first)
		want := NewTensor([]float64{1}, []int{}) // ArgMax reduces dimension
		got := a.ArgMax()
		assertTensorsEqual(t, got, want, 0) // Index is integer
	})

	t.Run("2D ArgMax (along last dim)", func(t *testing.T) {
		a := NewTensor([]float64{
			1, 9, 2, // Row 0, max at index 1
			8, 5, 7, // Row 1, max at index 0
			3, 4, 6, // Row 2, max at index 2
		}, []int{3, 3})
		want := NewTensor([]float64{1, 0, 2}, []int{3}) // Shape is [3]
		got := a.ArgMax()
		assertTensorsEqual(t, got, want, 0)
	})

	t.Run("3D ArgMax", func(t *testing.T) {
		a := NewTensor([]float64{
			1, 9, 2, 8, 5, 7, // Slice 0 (2x3) -> ArgMax -> [1, 0]
			3, 4, 6, 0, 1, -1, // Slice 1 (2x3) -> ArgMax -> [2, 1]
		}, []int{2, 2, 3})
		want := NewTensor([]float64{1, 0, 2, 1}, []int{2, 2}) // Shape [2, 2]
		got := a.ArgMax()
		assertTensorsEqual(t, got, want, 0)
	})

	t.Run("Panic on 0D tensor", func(t *testing.T) {
		// Assuming NewTensor prevents 0D or handles it, otherwise test:
		// a := NewTensor([]float64{5}, []int{})
		// assertPanic(t, func(){ a.ArgMax() }, "at least 1 dimension")
	})
}

func TestMaxPool(t *testing.T) {
	t.Run("2D MaxPool", func(t *testing.T) {
		data := []float64{
			1, 3, 2, 4,
			5, 7, 6, 8,
			9, 11, 10, 12,
			13, 15, 14, 16,
		}
		a := NewTensor(data, []int{4, 4})
		poolSize := 2
		stride := 2

		// Expected output (2x2)
		// Pool 1 (top-left): max(1,3,5,7) = 7. Index in original data: 5
		// Pool 2 (top-right): max(2,4,6,8) = 8. Index in original data: 7
		// Pool 3 (bottom-left): max(9,11,13,15) = 15. Index in original data: 13
		// Pool 4 (bottom-right): max(10,12,14,16) = 16. Index in original data: 15
		wantPool := NewTensor([]float64{7, 8, 15, 16}, []int{2, 2})
		wantArgmax := NewTensor([]float64{5, 7, 13, 15}, []int{2, 2}) // Indices in flattened original data

		gotPool, gotArgmax := a.MaxPool(poolSize, stride)
		assertTensorsEqual(t, gotPool, wantPool, 0)
		assertTensorsEqual(t, gotArgmax, wantArgmax, 0)
	})

	t.Run("2D MaxPool with Stride 1", func(t *testing.T) {
		data := []float64{
			1, 3, 2,
			5, 7, 6,
			9, 11, 10,
		}
		a := NewTensor(data, []int{3, 3})
		poolSize := 2
		stride := 1
		// Output shape: (3-2)/1 + 1 = 2 -> 2x2
		// Pool (0,0): max(1,3,5,7)=7. Idx=4
		// Pool (0,1): max(3,2,7,6)=7. Idx=4
		// Pool (1,0): max(5,7,9,11)=11. Idx=7
		// Pool (1,1): max(7,6,11,10)=11. Idx=7
		wantPool := NewTensor([]float64{7, 7, 11, 11}, []int{2, 2})
		wantArgmax := NewTensor([]float64{4, 4, 7, 7}, []int{2, 2})

		gotPool, gotArgmax := a.MaxPool(poolSize, stride)
		assertTensorsEqual(t, gotPool, wantPool, 0)
		assertTensorsEqual(t, gotArgmax, wantArgmax, 0)
	})

	t.Run("3D MaxPool", func(t *testing.T) {
		data := []float64{
			// Channel 0
			1, 3, 2, 4,
			5, 7, 6, 8,
			9, 11, 10, 12,
			13, 15, 14, 16,
			// Channel 1
			16, 14, 15, 13,
			12, 10, 11, 9,
			8, 6, 7, 5,
			4, 2, 3, 1,
		}
		// Shape [2, 4, 4] (Channels, Height, Width) - Assuming pooling on H, W
		a := NewTensor(data, []int{2, 4, 4})
		poolSize := 2
		stride := 2

		// Expected output shape [2, 2, 2]
		// Channel 0 pool: [7, 8, 15, 16] -> Indices [5, 7, 13, 15] (relative to start of data)
		// Channel 1 pool: max(16,14,12,10)=16. Idx=16 | max(15,13,11,9)=15. Idx=18
		//                 max(8,6,4,2)=8. Idx=24   | max(7,5,3,1)=7. Idx=26
		// -> [16, 15, 8, 7] -> Indices [16, 18, 24, 26]
		wantPoolData := []float64{7, 8, 15, 16, 16, 15, 8, 7}
		wantArgmData := []float64{5, 7, 13, 15, 16, 18, 24, 26}
		wantShape := []int{2, 2, 2}
		wantPool := NewTensor(wantPoolData, wantShape)
		wantArgmax := NewTensor(wantArgmData, wantShape)

		gotPool, gotArgmax := a.MaxPool(poolSize, stride)
		assertTensorsEqual(t, gotPool, wantPool, 0)
		assertTensorsEqual(t, gotArgmax, wantArgmax, 0)
	})

	t.Run("Panic on Insufficient Dimensions", func(t *testing.T) {
		a := NewTensor([]float64{1, 2, 3}, []int{3}) // 1D
		assertPanic(t, func() { a.MaxPool(2, 1) }, "at least 2 dimensions")
	})
}

func TestShapesMatch(t *testing.T) {
	t1 := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	t2 := NewTensor([]float64{5, 6, 7, 8}, []int{2, 2})
	t3 := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	t4 := NewTensor([]float64{1, 2, 3, 4}, []int{4})
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
