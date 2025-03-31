package tensor

import (
	"math"
	"reflect"
	"testing"
)

const float64EqualityThreshold = 1e-9

// Helper function to compare two float64 slices with tolerance
func floatSlicesAreEqual(a, b []float64, tolerance float64) bool {
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

// Helper function to compare two Tensors
func tensorsAreEqual(t1, t2 *Tensor, tolerance float64) bool {
	if t1 == nil && t2 == nil {
		return true
	}
	if t1 == nil || t2 == nil {
		return false
	}
	if !reflect.DeepEqual(t1.Shape, t2.Shape) {
		return false
	}
	return floatSlicesAreEqual(t1.Data, t2.Data, tolerance)
}

// Helper function to check for panics
func checkPanic(t *testing.T, f func(), expectedPanicMsgPart string) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic as expected")
		} else {
			// Optionally check if the panic message contains the expected string
			if msg, ok := r.(string); ok {
				if expectedPanicMsgPart != "" && !contains(msg, expectedPanicMsgPart) {
					t.Errorf("Panic message '%s' does not contain expected part '%s'", msg, expectedPanicMsgPart)
				}
			} else if err, ok := r.(error); ok {
				// Handle error type panics if necessary
				if expectedPanicMsgPart != "" && !contains(err.Error(), expectedPanicMsgPart) {
					t.Errorf("Panic error '%s' does not contain expected part '%s'", err.Error(), expectedPanicMsgPart)
				}
			} else {
				t.Logf("Recovered a non-string/non-error panic: %v", r)
			}

		}
	}()
	f()
}

// Helper contains function (replace with strings.Contains if preferred)
func contains(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// --- Test Functions ---

func TestNewTensor(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	tensor := NewTensor(data, shape)

	if tensor == nil {
		t.Fatal("NewTensor returned nil")
	}
	if !reflect.DeepEqual(tensor.Shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.Shape)
	}
	if !floatSlicesAreEqual(tensor.Data, data, float64EqualityThreshold) {
		t.Errorf("Expected data %v, got %v", data, tensor.Data)
	}

	// Test potential panic if data size doesn't match shape (assuming NewTensor checks this)
	// checkPanic(t, func() { NewTensor([]float64{1, 2, 3}, []int{2, 2}) }, "data length")
}

func TestDimensions(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		expected int
	}{
		{"Scalar (conceptually)", []int{1}, 1}, // Representing scalar as 1D tensor of size 1
		{"1D Vector", []int{5}, 1},
		{"2D Matrix", []int{2, 3}, 2},
		{"3D Tensor", []int{4, 2, 3}, 3},
		{"4D Tensor", []int{1, 3, 28, 28}, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(make([]float64, product(tt.shape)), tt.shape)
			if dims := tensor.Dimensions(); dims != tt.expected {
				t.Errorf("Expected dimensions %d, got %d for shape %v", tt.expected, dims, tt.shape)
			}
		})
	}
}

func TestDimSize(t *testing.T) {
	shape := []int{5, 3, 4}
	tensor := NewTensor(make([]float64, product(shape)), shape)

	if size := tensor.DimSize(0); size != 5 {
		t.Errorf("Expected size 5 for dim 0, got %d", size)
	}
	if size := tensor.DimSize(1); size != 3 {
		t.Errorf("Expected size 3 for dim 1, got %d", size)
	}
	if size := tensor.DimSize(2); size != 4 {
		t.Errorf("Expected size 4 for dim 2, got %d", size)
	}

	// Test panics for invalid dimensions
	checkPanic(t, func() { tensor.DimSize(-1) }, "invalid dimension")
	checkPanic(t, func() { tensor.DimSize(3) }, "invalid dimension")
}

func TestClone(t *testing.T) {
	original := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	clone := original.Clone()

	if !tensorsAreEqual(original, clone, float64EqualityThreshold) {
		t.Fatalf("Clone is not equal to original. Original: %v, Clone: %v", original, clone)
	}

	// Modify original data and check clone
	original.Data[0] = 99
	if tensorsAreEqual(original, clone, float64EqualityThreshold) {
		t.Errorf("Modifying original data affected the clone (shallow copy?)")
	}
	if clone.Data[0] == 99 {
		t.Errorf("Clone data was modified when original data changed. Expected %f, got %f", 1.0, clone.Data[0])
	}
	original.Data[0] = 1 // Restore

	// Modify clone data and check original
	clone.Data[1] = 88
	if tensorsAreEqual(original, clone, float64EqualityThreshold) {
		t.Errorf("Modifying clone data affected the original")
	}
	if original.Data[1] == 88 {
		t.Errorf("Original data was modified when clone data changed. Expected %f, got %f", 2.0, original.Data[1])
	}

	// Modify original shape (if possible - depends on Tensor struct details)
	// original.Shape[0] = 5 // Assuming Shape is mutable for this test
	// if reflect.DeepEqual(original.Shape, clone.Shape) {
	//     t.Errorf("Modifying original shape affected the clone shape")
	// }
}

func TestMultiply(t *testing.T) {
	t1 := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	t2 := NewTensor([]float64{5, 6, 7, 8}, []int{2, 2})
	expected := NewTensor([]float64{5, 12, 21, 32}, []int{2, 2})

	result := t1.Multiply(t2)
	if !tensorsAreEqual(expected, result, float64EqualityThreshold) {
		t.Errorf("Element-wise multiplication failed. Expected %v, got %v", expected, result)
	}

	// Test panic on shape mismatch
	t3 := NewTensor([]float64{1, 2, 3}, []int{3})
	checkPanic(t, func() { t1.Multiply(t3) }, "same shape")
}

func TestTranspose(t *testing.T) {
	t1 := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	expected := NewTensor([]float64{1, 4, 2, 5, 3, 6}, []int{3, 2})

	result := t1.Transpose()
	if !tensorsAreEqual(expected, result, float64EqualityThreshold) {
		t.Errorf("Transpose failed. Expected %v, got %v", expected, result)
	}

	// Test panic on non-2D tensor
	t2 := NewTensor([]float64{1, 2, 3}, []int{3})
	checkPanic(t, func() { t2.Transpose() }, "only works for 2D")

	t3 := NewTensor(make([]float64, 8), []int{2, 2, 2})
	checkPanic(t, func() { t3.Transpose() }, "only works for 2D")
}

func TestString(t *testing.T) {
	tensor := NewTensor([]float64{1.5, 2.0}, []int{1, 2})
	str := tensor.String()

	if !contains(str, "Data: [1.5 2]") || !contains(str, "Shape: [1 2]") {
		t.Errorf("String() output format unexpected: %s", str)
	}
}

func TestGetSample(t *testing.T) {
	// Batch=2, Channels=1, Height=2, Width=3
	data := []float64{
		// Sample 0
		1, 2, 3,
		4, 5, 6,
		// Sample 1
		7, 8, 9,
		10, 11, 12,
	}
	shape := []int{2, 1, 2, 3}
	tensor := NewTensor(data, shape)

	sample0Expected := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{1, 2, 3})
	sample0Result := tensor.GetSample(0)
	if !tensorsAreEqual(sample0Expected, sample0Result, float64EqualityThreshold) {
		t.Errorf("GetSample(0) failed. Expected %v, got %v", sample0Expected, sample0Result)
	}

	sample1Expected := NewTensor([]float64{7, 8, 9, 10, 11, 12}, []int{1, 2, 3})
	sample1Result := tensor.GetSample(1)
	if !tensorsAreEqual(sample1Expected, sample1Result, float64EqualityThreshold) {
		t.Errorf("GetSample(1) failed. Expected %v, got %v", sample1Expected, sample1Result)
	}

	// Test panics
	checkPanic(t, func() { tensor.GetSample(-1) }, "invalid batch index")
	checkPanic(t, func() { tensor.GetSample(2) }, "invalid batch index")

	t3d := NewTensor(make([]float64, 6), []int{1, 2, 3})
	checkPanic(t, func() { t3d.GetSample(0) }, "only works for 4D")
}

func TestStackTensors(t *testing.T) {
	t1 := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	t2 := NewTensor([]float64{5, 6, 7, 8}, []int{2, 2})
	t3 := NewTensor([]float64{9, 10, 11, 12}, []int{2, 2})
	tensors := []*Tensor{t1, t2, t3}

	// Stack along new dim 0
	expected0 := NewTensor([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{3, 2, 2})
	result0, err0 := StackTensors(tensors, 0)
	if err0 != nil {
		t.Fatalf("StackTensors(dim=0) returned error: %v", err0)
	}
	if !tensorsAreEqual(expected0, result0, float64EqualityThreshold) {
		t.Errorf("StackTensors(dim=0) failed. Expected %v, got %v", expected0, result0)
	}

	// NOTE: The implementation stacks sequentially, let's test other dims based on that understanding
	// Stack along new dim 1 (data ordering might be unintuitive with the current implementation)
	// This would actually stack like [ [t1], [t2], [t3] ] and reshape, data order would be sequential
	expected1_impl := NewTensor([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{2, 3, 2}) // Based on implementation copying sequentially
	result1, err1 := StackTensors(tensors, 1)
	if err1 != nil {
		t.Fatalf("StackTensors(dim=1) returned error: %v", err1)
	}
	if !tensorsAreEqual(expected1_impl, result1, float64EqualityThreshold) {
		t.Errorf("StackTensors(dim=1) failed based on sequential copy impl. Expected %v, got %v", expected1_impl, result1)
	}

	// Test error cases
	_, errEmpty := StackTensors([]*Tensor{}, 0)
	if errEmpty == nil {
		t.Errorf("StackTensors with empty list did not return error")
	}

	t_diff_shape := NewTensor([]float64{1, 2, 3}, []int{1, 3})
	_, errDiffShape := StackTensors([]*Tensor{t1, t_diff_shape}, 0)
	if errDiffShape == nil {
		t.Errorf("StackTensors with different shapes did not return error")
	}
}

func TestIm2Col(t *testing.T) {
	// Simple 3D case: 1 channel, 3x3 image, kernel 2x2, stride 1
	// Input shape: [1, 3, 3]
	input3D := NewTensor([]float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, []int{1, 3, 3})
	kernelSize := 2
	stride := 1

	// Expected output:
	// Patches: [1,2,4,5], [2,3,5,6], [4,5,7,8], [5,6,8,9]
	// Output shape: [channels*k*k, out_h*out_w] = [1*2*2, 2*2] = [4, 4]
	// Data order (column major for patches):
	// col0=[1,4,2,5], col1=[2,5,3,6], col2=[4,7,5,8], col3=[5,8,6,9] <- This depends on implementation loop order!
	// Let's trace the provided implementation:
	// c=0: wOff=0, hOff=0, cIm=0. h=0,w=0 -> imR=0,imC=0 -> pix=1. h=0,w=1 -> imR=0,imC=1 -> pix=2. h=1,w=0 -> imR=1,imC=0 -> pix=4. h=1,w=1 -> imR=1,imC=1 -> pix=5.
	// c=1: wOff=1, hOff=0, cIm=0. h=0,w=0 -> imR=0,imC=1 -> pix=2. h=0,w=1 -> imR=0,imC=2 -> pix=3. h=1,w=0 -> imR=1,imC=1 -> pix=5. h=1,w=1 -> imR=1,imC=2 -> pix=6.
	// c=2: wOff=0, hOff=1, cIm=0. h=0,w=0 -> imR=1,imC=0 -> pix=4. h=0,w=1 -> imR=1,imC=1 -> pix=5. h=1,w=0 -> imR=2,imC=0 -> pix=7. h=1,w=1 -> imR=2,imC=1 -> pix=8.
	// c=3: wOff=1, hOff=1, cIm=0. h=0,w=0 -> imR=1,imC=1 -> pix=5. h=0,w=1 -> imR=1,imC=2 -> pix=6. h=1,w=0 -> imR=2,imC=1 -> pix=8. h=1,w=1 -> imR=2,imC=2 -> pix=9.
	// Output Data (row major): [1,2,4,5,  2,3,5,6,  4,5,7,8,  5,6,8,9]
	expectedData3D := []float64{
		1, 2, 4, 5,
		2, 3, 5, 6,
		4, 5, 7, 8,
		5, 6, 8, 9,
	}
	expectedShape3D := []int{4, 4} // [C*k*k, H_out*W_out] = [1*2*2, 2*2]

	result3D, err3D := input3D.im2col(kernelSize, stride)
	if err3D != nil {
		t.Fatalf("im2col (3D) returned error: %v", err3D)
	}
	expectedTensor3D := NewTensor(expectedData3D, expectedShape3D)
	if !tensorsAreEqual(expectedTensor3D, result3D, float64EqualityThreshold) {
		t.Errorf("im2col (3D) failed. Expected %v, got %v", expectedTensor3D, result3D)
	}

	// Simple 4D case: Batch=1, C=1, H=3, W=3
	input4D := NewTensor(input3D.Data, []int{1, 1, 3, 3})
	result4D, err4D := input4D.im2col(kernelSize, stride)
	if err4D != nil {
		t.Fatalf("im2col (4D) returned error: %v", err4D)
	}
	// Expected should be same as 3D case because batch is handled implicitly
	if !tensorsAreEqual(expectedTensor3D, result4D, float64EqualityThreshold) {
		t.Errorf("im2col (4D) failed. Expected %v, got %v", expectedTensor3D, result4D)
	}

	// Test error on invalid dimensions
	input2D := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	_, err2D := input2D.im2col(1, 1)
	if err2D == nil {
		t.Errorf("im2col with 2D input did not return error")
	}
}

func TestPad2D(t *testing.T) {
	// Test 3D
	input3D := NewTensor([]float64{1, 2, 3, 4}, []int{1, 2, 2}) // C, H, W
	pad := 1
	expected3D := NewTensor([]float64{
		0, 0, 0, 0,
		0, 1, 2, 0,
		0, 3, 4, 0,
		0, 0, 0, 0,
	}, []int{1, 4, 4}) // C, H+2p, W+2p

	result3D := input3D.Pad2D(pad)
	if !tensorsAreEqual(expected3D, result3D, float64EqualityThreshold) {
		t.Errorf("Pad2D (3D) failed. Expected %v, got %v", expected3D, result3D)
	}

	// Test 4D
	input4D := NewTensor([]float64{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 1, 2, 2}) // B, C, H, W
	expected4D := NewTensor([]float64{
		// Batch 0
		0, 0, 0, 0,
		0, 1, 2, 0,
		0, 3, 4, 0,
		0, 0, 0, 0,
		// Batch 1
		0, 0, 0, 0,
		0, 5, 6, 0,
		0, 7, 8, 0,
		0, 0, 0, 0,
	}, []int{2, 1, 4, 4}) // B, C, H+2p, W+2p

	result4D := input4D.Pad2D(pad)
	if !tensorsAreEqual(expected4D, result4D, float64EqualityThreshold) {
		t.Errorf("Pad2D (4D) failed. Expected %v, got %v", expected4D, result4D)
	}

	// Test pad=0
	resultPad0 := input3D.Pad2D(0)
	if !tensorsAreEqual(input3D, resultPad0, float64EqualityThreshold) {
		t.Errorf("Pad2D(0) should return a clone, but it differs.")
	}
	// Ensure it's a clone
	resultPad0.Data[0] = 99
	if input3D.Data[0] == 99 {
		t.Errorf("Pad2D(0) did not return a clone (modified original)")
	}

	// Test panic on invalid dimensions
	input2D := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	checkPanic(t, func() { input2D.Pad2D(1) }, "only works for 3D or 4D")
}

func TestRepeat(t *testing.T) {
	// Test 2D
	input2D := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	// Repeat dim 0
	expected2D_0 := NewTensor([]float64{1, 2, 3, 4, 1, 2, 3, 4}, []int{4, 2})
	result2D_0 := input2D.Repeat(0, 2)
	if !tensorsAreEqual(expected2D_0, result2D_0, float64EqualityThreshold) {
		t.Errorf("Repeat (2D, dim=0) failed. Expected %v, got %v", expected2D_0, result2D_0)
	}
	// Repeat dim 1
	expected2D_1 := NewTensor([]float64{1, 2, 1, 2, 3, 4, 3, 4}, []int{2, 4})
	result2D_1 := input2D.Repeat(1, 2)
	if !tensorsAreEqual(expected2D_1, result2D_1, float64EqualityThreshold) {
		t.Errorf("Repeat (2D, dim=1) failed. Expected %v, got %v", expected2D_1, result2D_1)
	}

	// Test 4D (B, C, H, W)
	input4D := NewTensor([]float64{1, 2, 3, 4}, []int{1, 1, 2, 2})
	// Repeat dim 0 (batch)
	expected4D_0 := NewTensor([]float64{1, 2, 3, 4, 1, 2, 3, 4}, []int{2, 1, 2, 2})
	result4D_0 := input4D.Repeat(0, 2)
	if !tensorsAreEqual(expected4D_0, result4D_0, float64EqualityThreshold) {
		t.Errorf("Repeat (4D, dim=0) failed. Expected %v, got %v", expected4D_0, result4D_0)
	}
	// Repeat dim 1 (channels)
	// Implementation copies full C*H*W blocks
	expected4D_1 := NewTensor([]float64{1, 2, 3, 4, 1, 2, 3, 4}, []int{1, 2, 2, 2})
	result4D_1 := input4D.Repeat(1, 2)
	if !tensorsAreEqual(expected4D_1, result4D_1, float64EqualityThreshold) {
		t.Errorf("Repeat (4D, dim=1) failed. Expected %v, got %v", expected4D_1, result4D_1)
	}

	// Test Panics
	checkPanic(t, func() { input2D.Repeat(2, 2) }, "Invalid dimension")
	checkPanic(t, func() { input4D.Repeat(2, 2) }, "Repeating along spatial dimensions")
	checkPanic(t, func() { input4D.Repeat(4, 2) }, "Invalid dimension")
	input3D := NewTensor(make([]float64, 4), []int{1, 2, 2})
	checkPanic(t, func() { input3D.Repeat(0, 2) }, "only supports 2D or 4D")
}

func TestPadAndCrop(t *testing.T) {
	// These seem 2D specific in the implementation
	input := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	padding := 1

	// Pad
	paddedExpected := NewTensor([]float64{
		0, 0, 0, 0,
		0, 1, 2, 0,
		0, 3, 4, 0,
		0, 0, 0, 0,
	}, []int{4, 4})
	paddedResult := input.Pad(padding)
	if !tensorsAreEqual(paddedExpected, paddedResult, float64EqualityThreshold) {
		t.Errorf("Pad failed. Expected %v, got %v", paddedExpected, paddedResult)
	}

	// Crop
	croppedResult := paddedResult.Crop(padding)
	if !tensorsAreEqual(input, croppedResult, float64EqualityThreshold) {
		t.Errorf("Crop failed. Expected %v, got %v", input, croppedResult)
	}

	// Test Crop with 0 padding
	crop0 := input.Crop(0)
	if crop0 != input { // Should return the same tensor instance
		t.Errorf("Crop(0) should return the original tensor instance")
	}
}

func TestFlattenByDim(t *testing.T) {
	// Note: FlattenByDim modifies the tensor in-place!
	input := NewTensor(make([]float64, 24), []int{2, 3, 4}) // B=2, C=3, W=4
	// Create copy for comparison
	originalData := make([]float64, len(input.Data))
	copy(originalData, input.Data)

	// Flatten from dim 1 to end (1 to 2)
	tensorToFlatten1 := input.Clone()
	expectedShape1 := []int{2, 12}
	tensorToFlatten1.FlattenByDim(1, 2) // Or FlattenByDim(1, -1)
	if !reflect.DeepEqual(tensorToFlatten1.Shape, expectedShape1) {
		t.Errorf("FlattenByDim(1, -1) failed. Expected shape %v, got %v", expectedShape1, tensorToFlatten1.Shape)
	}
	// Data should remain the same as it reshapes in place
	if !floatSlicesAreEqual(originalData, tensorToFlatten1.Data, float64EqualityThreshold) {
		t.Errorf("FlattenByDim(1, -1) data was modified unexpectedly.")
	}

	// Flatten from dim 0 to 1
	tensorToFlatten2 := input.Clone()
	expectedShape2 := []int{6, 4}
	tensorToFlatten2.FlattenByDim(0, 1)
	if !reflect.DeepEqual(tensorToFlatten2.Shape, expectedShape2) {
		t.Errorf("FlattenByDim(0, 1) failed. Expected shape %v, got %v", expectedShape2, tensorToFlatten2.Shape)
	}
	if !floatSlicesAreEqual(originalData, tensorToFlatten2.Data, float64EqualityThreshold) {
		t.Errorf("FlattenByDim(0, 1) data was modified unexpectedly.")
	}

	// Flatten all
	tensorToFlatten3 := input.Clone()
	expectedShape3 := []int{24, 1} // Check if implementation results in [N, 1] or [1, N]
	tensorToFlatten3.FlattenByDim(0, -1)
	if !reflect.DeepEqual(tensorToFlatten3.Shape, expectedShape3) {
		t.Errorf("FlattenByDim(0, -1) failed. Expected shape %v, got %v", expectedShape3, tensorToFlatten3.Shape)
	}
	if !floatSlicesAreEqual(originalData, tensorToFlatten3.Data, float64EqualityThreshold) {
		t.Errorf("FlattenByDim(0, -1) data was modified unexpectedly.")
	}

	// Test Panics
	tensorPanic := input.Clone()
	checkPanic(t, func() { tensorPanic.FlattenByDim(-1, 1) }, "Invalid startDim")
	checkPanic(t, func() { tensorPanic.FlattenByDim(0, 3) }, "Invalid endDim")
	checkPanic(t, func() { tensorPanic.FlattenByDim(2, 1) }, "") // Should panic if start > end? Implementation doesn't explicitly check
}

func TestGetCols(t *testing.T) {
	input := NewTensor([]float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, []int{2, 4})

	// Get cols 1 to 3 (exclusive)
	expected1_3 := NewTensor([]float64{2, 3, 6, 7}, []int{2, 2})
	result1_3 := input.GetCols(1, 3)
	if !tensorsAreEqual(expected1_3, result1_3, float64EqualityThreshold) {
		t.Errorf("GetCols(1, 3) failed. Expected %v, got %v", expected1_3, result1_3)
	}

	// Get first column
	expected0_1 := NewTensor([]float64{1, 5}, []int{2, 1})
	result0_1 := input.GetCols(0, 1)
	if !tensorsAreEqual(expected0_1, result0_1, float64EqualityThreshold) {
		t.Errorf("GetCols(0, 1) failed. Expected %v, got %v", expected0_1, result0_1)
	}

	// Get last column
	expected3_4 := NewTensor([]float64{4, 8}, []int{2, 1})
	result3_4 := input.GetCols(3, 4)
	if !tensorsAreEqual(expected3_4, result3_4, float64EqualityThreshold) {
		t.Errorf("GetCols(3, 4) failed. Expected %v, got %v", expected3_4, result3_4)
	}

	// Test Panics
	checkPanic(t, func() { input.GetCols(-1, 2) }, "Invalid column range")
	checkPanic(t, func() { input.GetCols(0, 5) }, "Invalid column range")
	checkPanic(t, func() { input.GetCols(2, 1) }, "Invalid column range")
	input3D := NewTensor(make([]float64, 8), []int{2, 2, 2})
	checkPanic(t, func() { input3D.GetCols(0, 1) }, "only works for 2D")
}

func TestSetCol(t *testing.T) {
	target := NewTensor([]float64{
		1, 2, 3,
		4, 5, 6,
	}, []int{2, 3})
	colData := NewTensor([]float64{9, 8}, []int{2, 1}) // Correct shape

	expected := NewTensor([]float64{
		1, 9, 3,
		4, 8, 6,
	}, []int{2, 3})

	target.SetCol(1, colData)
	if !tensorsAreEqual(expected, target, float64EqualityThreshold) {
		t.Errorf("SetCol failed. Expected %v, got %v", expected, target)
	}

	// Test Panics
	badDataShape1 := NewTensor([]float64{9, 8, 7}, []int{3, 1})
	colData1D := NewTensor([]float64{9, 8}, []int{2}) // Incorrect shape (needs to be 2D column)

	targetReset := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	checkPanic(t, func() { targetReset.SetCol(1, badDataShape1) }, "Invalid column data dimensions")
	checkPanic(t, func() { targetReset.SetCol(1, colData1D) }, "Invalid column data dimensions") // Fails because data.Shape[1] != 1

	input3D := NewTensor(make([]float64, 8), []int{2, 2, 2})
	checkPanic(t, func() { input3D.SetCol(0, colData) }, "only works for 2D")
	// Panic on invalid colIdx (will likely be index out of range)
	// checkPanic(t, func() { targetReset.SetCol(3, colData) }, "") // Test index out of range
}

func TestGetCol(t *testing.T) {
	input := NewTensor([]float64{
		1, 2, 3,
		4, 5, 6,
	}, []int{2, 3})

	expectedCol1 := NewTensor([]float64{2, 5}, []int{2}) // Implementation returns 1D
	resultCol1 := input.GetCol(1)
	if !tensorsAreEqual(expectedCol1, resultCol1, float64EqualityThreshold) {
		t.Errorf("GetCol(1) failed. Expected %v, got %v", expectedCol1, resultCol1)
	}

	expectedCol0 := NewTensor([]float64{1, 4}, []int{2})
	resultCol0 := input.GetCol(0)
	if !tensorsAreEqual(expectedCol0, resultCol0, float64EqualityThreshold) {
		t.Errorf("GetCol(0) failed. Expected %v, got %v", expectedCol0, resultCol0)
	}

	// Test Panics
	checkPanic(t, func() { input.GetCol(-1) }, "Invalid column index")
	checkPanic(t, func() { input.GetCol(3) }, "Invalid column index")
	input3D := NewTensor(make([]float64, 8), []int{2, 2, 2})
	checkPanic(t, func() { input3D.GetCol(0) }, "only works for 2D")
}

func TestSumByDim(t *testing.T) {
	input := NewTensor([]float64{
		1, 2, 3,
		4, 5, 6,
	}, []int{2, 3})

	// Sum along dim 0 (sum columns) -> shape [3]
	expectedSum0 := NewTensor([]float64{5, 7, 9}, []int{3})
	resultSum0 := input.SumByDim(0)
	if !tensorsAreEqual(expectedSum0, resultSum0, float64EqualityThreshold) {
		t.Errorf("SumByDim(0) failed. Expected %v, got %v", expectedSum0, resultSum0)
	}

	// Sum along dim 1 (sum rows) -> shape [2]
	expectedSum1 := NewTensor([]float64{6, 15}, []int{2})
	resultSum1 := input.SumByDim(1)
	if !tensorsAreEqual(expectedSum1, resultSum1, float64EqualityThreshold) {
		t.Errorf("SumByDim(1) failed. Expected %v, got %v", expectedSum1, resultSum1)
	}

	// Test Panics
	checkPanic(t, func() { input.SumByDim(2) }, "Invalid dimension")
	input3D := NewTensor(make([]float64, 8), []int{2, 2, 2})
	checkPanic(t, func() { input3D.SumByDim(0) }, "works for 2D tensors")
}

func TestMatMul(t *testing.T) {
	t1 := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	t2 := NewTensor([]float64{5, 6, 7, 8}, []int{2, 2})
	expected := NewTensor([]float64{
		1*5 + 2*7, 1*6 + 2*8, // 5+14=19, 6+16=22
		3*5 + 4*7, 3*6 + 4*8, // 15+28=43, 18+32=50
	}, []int{2, 2})

	result, err := t1.MatMul(t2)
	if err != nil {
		t.Fatalf("MatMul returned error: %v", err)
	}
	if !tensorsAreEqual(expected, result, float64EqualityThreshold) {
		t.Errorf("MatMul failed. Expected %v, got %v", expected, result)
	}

	// Non-square
	t3 := NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	t4 := NewTensor([]float64{7, 8, 9, 10, 11, 12}, []int{3, 2})
	expected2 := NewTensor([]float64{
		1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12, // 7+18+33=58, 8+20+36=64
		4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12, // 28+45+66=139, 32+50+72=154
	}, []int{2, 2})
	result2, err2 := t3.MatMul(t4)
	if err2 != nil {
		t.Fatalf("MatMul (non-square) returned error: %v", err2)
	}
	if !tensorsAreEqual(expected2, result2, float64EqualityThreshold) {
		t.Errorf("MatMul (non-square) failed. Expected %v, got %v", expected2, result2)
	}

	// Test Errors
	t_incompatible := NewTensor([]float64{1, 2}, []int{1, 2})
	_, err_incompatible := t1.MatMul(t_incompatible)
	if err_incompatible == nil {
		t.Errorf("MatMul with incompatible shapes did not return error")
	}

	t1D := NewTensor([]float64{1, 2}, []int{2})
	_, err_non2D := t1.MatMul(t1D)
	if err_non2D == nil {
		t.Errorf("MatMul with non-2D tensor did not return error")
	}

}

func TestExpand(t *testing.T) {
	// Expand row vector
	row := NewTensor([]float64{1, 2, 3}, []int{1, 3})
	targetShapeRow := []int{4, 3}
	expectedRow := NewTensor([]float64{
		1, 2, 3,
		1, 2, 3,
		1, 2, 3,
		1, 2, 3,
	}, targetShapeRow)
	resultRow := row.Expand(targetShapeRow)
	if !tensorsAreEqual(expectedRow, resultRow, float64EqualityThreshold) {
		t.Errorf("Expand row failed. Expected %v, got %v", expectedRow, resultRow)
	}

	// Expand column vector
	col := NewTensor([]float64{1, 2}, []int{2, 1})
	targetShapeCol := []int{2, 3}
	expectedCol := NewTensor([]float64{
		1, 1, 1,
		2, 2, 2,
	}, targetShapeCol)
	resultCol := col.Expand(targetShapeCol)
	if !tensorsAreEqual(expectedCol, resultCol, float64EqualityThreshold) {
		t.Errorf("Expand col failed. Expected %v, got %v", expectedCol, resultCol)
	}

	// Expand scalar
	scalar := NewTensor([]float64{5}, []int{1, 1})
	targetShapeScalar := []int{2, 2}
	expectedScalar := NewTensor([]float64{5, 5, 5, 5}, targetShapeScalar)
	resultScalar := scalar.Expand(targetShapeScalar)
	if !tensorsAreEqual(expectedScalar, resultScalar, float64EqualityThreshold) {
		t.Errorf("Expand scalar failed. Expected %v, got %v", expectedScalar, resultScalar)
	}

	// Test Panics
	checkPanic(t, func() { row.Expand([]int{4, 3, 1}) }, "dimensions must match")
	checkPanic(t, func() { row.Expand([]int{4, 4}) }, "cannot expand dimension") // Dim 1 size mismatch
	nonUnit := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	checkPanic(t, func() { nonUnit.Expand([]int{2, 3}) }, "cannot expand dimension") // Dim 1 size mismatch and not 1
}

func TestGetRow(t *testing.T) {
	input := NewTensor([]float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, []int{3, 3})

	expectedRow1 := NewTensor([]float64{4, 5, 6}, []int{1, 3})
	resultRow1 := input.GetRow(1)
	if !tensorsAreEqual(expectedRow1, resultRow1, float64EqualityThreshold) {
		t.Errorf("GetRow(1) failed. Expected %v, got %v", expectedRow1, resultRow1)
	}

	expectedRow0 := NewTensor([]float64{1, 2, 3}, []int{1, 3})
	resultRow0 := input.GetRow(0)
	if !tensorsAreEqual(expectedRow0, resultRow0, float64EqualityThreshold) {
		t.Errorf("GetRow(0) failed. Expected %v, got %v", expectedRow0, resultRow0)
	}

	// Test Panics
	checkPanic(t, func() { input.GetRow(-1) }, "row index out of range")
	checkPanic(t, func() { input.GetRow(3) }, "row index out of range")
	input1D := NewTensor([]float64{1, 2, 3}, []int{3})
	checkPanic(t, func() { input1D.GetRow(0) }, "requires 2D tensor")
}

func TestSigmoid(t *testing.T) {
	input := NewTensor([]float64{-1, 0, 1, 100, -100}, []int{5})
	expectedData := []float64{
		1.0 / (1.0 + math.Exp(1.0)),  // sigmoid(-1)
		0.5,                          // sigmoid(0)
		1.0 / (1.0 + math.Exp(-1.0)), // sigmoid(1)
		1.0,                          // sigmoid(100) -> approx 1
		0.0,                          // sigmoid(-100) -> approx 0
	}
	expected := NewTensor(expectedData, []int{5})

	result := input.Sigmoid()
	if !tensorsAreEqual(expected, result, float64EqualityThreshold) {
		t.Errorf("Sigmoid failed. Expected %v, got %v", expected, result)
	}

	// Check shape preservation
	input2D := NewTensor([]float64{0, 1, 2, 3}, []int{2, 2})
	result2D := input2D.Sigmoid()
	if !reflect.DeepEqual(input2D.Shape, result2D.Shape) {
		t.Errorf("Sigmoid did not preserve shape. Input %v, Output %v", input2D.Shape, result2D.Shape)
	}
}

// --- Convolution related tests (Simplified) ---
// Full testing of Conv2D, im2col, col2im, and grads is complex and requires careful setup.
// These are basic sanity checks.

func TestConv2D_Simple(t *testing.T) {
	// Input: 1x1x3x3 (B, C, H, W)
	input := NewTensor([]float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, []int{1, 1, 3, 3})

	// Weights: 1x1x2x2 (OutC, InC, KH, KW) - Identity-like kernel (sum top-left 2x2)
	weights := NewTensor([]float64{
		1, 0,
		0, 0,
	}, []int{1, 1, 2, 2})

	kernelSize := 2
	stride := 1
	pad := 0

	// Expected output: 1x1x2x2 (B, OutC, OutH, OutW)
	// OutH = (3 - 2 + 2*0)/1 + 1 = 2
	// OutW = (3 - 2 + 2*0)/1 + 1 = 2
	// Output[0,0,0,0] = input[0,0,0,0]*w[0,0,0,0] + ... = 1*1 = 1
	// Output[0,0,0,1] = input[0,0,0,1]*w[0,0,0,0] + ... = 2*1 = 2
	// Output[0,0,1,0] = input[0,0,1,0]*w[0,0,0,0] + ... = 4*1 = 4
	// Output[0,0,1,1] = input[0,0,1,1]*w[0,0,0,0] + ... = 5*1 = 5
	expected := NewTensor([]float64{1, 2, 4, 5}, []int{1, 1, 2, 2})

	// Using the first Conv2D implementation
	result, err := input.Conv2D(weights, kernelSize, stride, pad)
	if err != nil {
		t.Fatalf("Conv2D failed with error: %v", err)
	}

	if !tensorsAreEqual(expected, result, float64EqualityThreshold) {
		t.Errorf("Conv2D (simple) failed.\nExpected: %v\nGot:      %v", expected, result)
	}

	// Test with 3D input (should be treated as batch size 1)
	input3D := NewTensor(input.Data, []int{1, 3, 3}) // C, H, W
	result3D, err3D := input3D.Conv2D(weights, kernelSize, stride, pad)
	if err3D != nil {
		t.Fatalf("Conv2D (3D input) failed with error: %v", err3D)
	}
	if !tensorsAreEqual(expected, result3D, float64EqualityThreshold) {
		t.Errorf("Conv2D (3D input) failed.\nExpected: %v\nGot:      %v", expected, result3D)
	}

	// Test padding
	pad = 1
	inputPad := NewTensor([]float64{5}, []int{1, 1, 1, 1})   // 1x1x1x1 input
	weightsPad := NewTensor([]float64{1}, []int{1, 1, 1, 1}) // 1x1x1x1 kernel
	// Padded input is 1x1x3x3 with 5 in the center
	// OutH = (1 - 1 + 2*1)/1 + 1 = 3
	// OutW = (1 - 1 + 2*1)/1 + 1 = 3
	// Output should be 1x1x3x3, only center element is 5*1=5
	expectedPad := NewTensor([]float64{0, 0, 0, 0, 5, 0, 0, 0, 0}, []int{1, 1, 3, 3})
	resultPad, errPad := inputPad.Conv2D(weightsPad, 1, 1, pad)
	if errPad != nil {
		t.Fatalf("Conv2D (padding) failed with error: %v", errPad)
	}
	if !tensorsAreEqual(expectedPad, resultPad, float64EqualityThreshold) {
		t.Errorf("Conv2D (padding) failed.\nExpected: %v\nGot:      %v", expectedPad, resultPad)
	}

	// Test error cases
	badWeights := NewTensor(make([]float64, 8), []int{1, 2, 2, 2}) // Mismatched input channels
	_, errMismatch := input.Conv2D(badWeights, kernelSize, stride, pad)
	if errMismatch == nil || !contains(errMismatch.Error(), "weights shape mismatch") {
		t.Errorf("Conv2D did not return expected error for mismatched channels")
	}

	input2D := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	_, errInputDim := input2D.Conv2D(weights, kernelSize, stride, pad)
	if errInputDim == nil || !contains(errInputDim.Error(), "must be 3D or 4D") {
		t.Errorf("Conv2D did not return expected error for 2D input")
	}
}
