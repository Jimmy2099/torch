package tensor

import (
	math "github.com/chewxy/math32"
	"reflect"
	"testing"
)

const float32EqualityThreshold = 1e-9

func floatSlicesAreEqual(a, b []float32, tolerance float32) bool {
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

func tensorsAreEqual(t1, t2 *Tensor, tolerance float32) bool {
	if t1 == nil && t2 == nil {
		return true
	}
	if t1 == nil || t2 == nil {
		return false
	}
	if !reflect.DeepEqual(t1.shape, t2.shape) {
		return false
	}
	return floatSlicesAreEqual(t1.Data, t2.Data, tolerance)
}

func checkPanic(t *testing.T, f func(), expectedPanicMsgPart string) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic as expected")
		} else {
			if msg, ok := r.(string); ok {
				if expectedPanicMsgPart != "" && !contains(msg, expectedPanicMsgPart) {
					t.Errorf("Panic message '%s' does not contain expected part '%s'", msg, expectedPanicMsgPart)
				}
			} else if err, ok := r.(error); ok {
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

func contains(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestNewTensor(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	tensor := NewTensor(data, shape)

	if tensor == nil {
		t.Fatal("NewGraphTensor returned nil")
	}
	if !reflect.DeepEqual(tensor.shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.shape)
	}
	if !floatSlicesAreEqual(tensor.Data, data, float32EqualityThreshold) {
		t.Errorf("Expected data %v, got %v", data, tensor.Data)
	}

}

func TestDimensions(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		expected int
	}{
		{"Scalar (conceptually)", []int{1}, 1},
		{"1D Vector", []int{5}, 1},
		{"2D Matrix", []int{2, 3}, 2},
		{"3D Tensor", []int{4, 2, 3}, 3},
		{"4D Tensor", []int{1, 3, 28, 28}, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(make([]float32, product(tt.shape)), tt.shape)
			if dims := tensor.Dimensions(); dims != tt.expected {
				t.Errorf("Expected dimensions %d, got %d for shape %v", tt.expected, dims, tt.shape)
			}
		})
	}
}

func TestDimSize(t *testing.T) {
	shape := []int{5, 3, 4}
	tensor := NewTensor(make([]float32, product(shape)), shape)

	if size := tensor.DimSize(0); size != 5 {
		t.Errorf("Expected size 5 for dim 0, got %d", size)
	}
	if size := tensor.DimSize(1); size != 3 {
		t.Errorf("Expected size 3 for dim 1, got %d", size)
	}
	if size := tensor.DimSize(2); size != 4 {
		t.Errorf("Expected size 4 for dim 2, got %d", size)
	}

	checkPanic(t, func() { tensor.DimSize(-1) }, "invalid dimension")
	checkPanic(t, func() { tensor.DimSize(3) }, "invalid dimension")
}

func TestClone(t *testing.T) {
	original := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	clone := original.Clone()

	if !tensorsAreEqual(original, clone, float32EqualityThreshold) {
		t.Fatalf("Clone is not equal to original. Original: %v, Clone: %v", original, clone)
	}

	original.Data[0] = 99
	if tensorsAreEqual(original, clone, float32EqualityThreshold) {
		t.Errorf("Modifying original data affected the clone (shallow copy?)")
	}
	if clone.Data[0] == 99 {
		t.Errorf("Clone data was modified when original data changed. Expected %f, got %f", 1.0, clone.Data[0])
	}
	original.Data[0] = 1

	clone.Data[1] = 88
	if tensorsAreEqual(original, clone, float32EqualityThreshold) {
		t.Errorf("Modifying clone data affected the original")
	}
	if original.Data[1] == 88 {
		t.Errorf("Original data was modified when clone data changed. Expected %f, got %f", 2.0, original.Data[1])
	}

}

func TestMultiply(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	t2 := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	expected := NewTensor([]float32{5, 12, 21, 32}, []int{2, 2})

	result := t1.Multiply(t2)
	if !tensorsAreEqual(expected, result, float32EqualityThreshold) {
		t.Errorf("Element-wise multiplication failed. Expected %v, got %v", expected, result)
	}

	t3 := NewTensor([]float32{1, 2, 3}, []int{3})
	checkPanic(t, func() { t1.Multiply(t3) }, "same shape")
}

func TestTranspose(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	expected := NewTensor([]float32{1, 4, 2, 5, 3, 6}, []int{3, 2})

	result := t1.Transpose()
	if !tensorsAreEqual(expected, result, float32EqualityThreshold) {
		t.Errorf("Transpose failed. Expected %v, got %v", expected, result)
	}

	t2 := NewTensor([]float32{1, 2, 3}, []int{3})
	checkPanic(t, func() { t2.Transpose() }, "only works for 2D")

	t3 := NewTensor(make([]float32, 8), []int{2, 2, 2})
	checkPanic(t, func() { t3.Transpose() }, "only works for 2D")
}

func TestString(t *testing.T) {
	tensor := NewTensor([]float32{1.5, 2.0}, []int{1, 2})
	str := tensor.String()

	if !contains(str, "Data: [1.5 2]") || !contains(str, "shape: [1 2]") {
		t.Errorf("String() output format unexpected: %s", str)
	}
}

func TestGetSample(t *testing.T) {
	data := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}
	shape := []int{2, 1, 2, 3}
	tensor := NewTensor(data, shape)

	sample0Expected := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{1, 2, 3})
	sample0Result := tensor.GetSample(0)
	if !tensorsAreEqual(sample0Expected, sample0Result, float32EqualityThreshold) {
		t.Errorf("GetSample(0) failed. Expected %v, got %v", sample0Expected, sample0Result)
	}

	sample1Expected := NewTensor([]float32{7, 8, 9, 10, 11, 12}, []int{1, 2, 3})
	sample1Result := tensor.GetSample(1)
	if !tensorsAreEqual(sample1Expected, sample1Result, float32EqualityThreshold) {
		t.Errorf("GetSample(1) failed. Expected %v, got %v", sample1Expected, sample1Result)
	}

	checkPanic(t, func() { tensor.GetSample(-1) }, "invalid batch index")
	checkPanic(t, func() { tensor.GetSample(2) }, "invalid batch index")

	t3d := NewTensor(make([]float32, 6), []int{1, 2, 3})
	checkPanic(t, func() { t3d.GetSample(0) }, "only works for 4D")
}

func TestStackTensors(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	t2 := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	t3 := NewTensor([]float32{9, 10, 11, 12}, []int{2, 2})
	tensors := []*Tensor{t1, t2, t3}

	expected0 := NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{3, 2, 2})
	result0, err0 := StackTensors(tensors, 0)
	if err0 != nil {
		t.Fatalf("StackTensors(dim=0) returned error: %v", err0)
	}
	if !tensorsAreEqual(expected0, result0, float32EqualityThreshold) {
		t.Errorf("StackTensors(dim=0) failed. Expected %v, got %v", expected0, result0)
	}

	expected1_impl := NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{2, 3, 2})
	result1, err1 := StackTensors(tensors, 1)
	if err1 != nil {
		t.Fatalf("StackTensors(dim=1) returned error: %v", err1)
	}
	if !tensorsAreEqual(expected1_impl, result1, float32EqualityThreshold) {
		t.Errorf("StackTensors(dim=1) failed based on sequential copy impl. Expected %v, got %v", expected1_impl, result1)
	}

	_, errEmpty := StackTensors([]*Tensor{}, 0)
	if errEmpty == nil {
		t.Errorf("StackTensors with empty list did not return error")
	}

	t_diff_shape := NewTensor([]float32{1, 2, 3}, []int{1, 3})
	_, errDiffShape := StackTensors([]*Tensor{t1, t_diff_shape}, 0)
	if errDiffShape == nil {
		t.Errorf("StackTensors with different shapes did not return error")
	}
}

func TestIm2Col(t *testing.T) {
	input3D := NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, []int{1, 3, 3})
	kernelSize := 2
	stride := 1

	expectedData3D := []float32{
		1, 2, 4, 5,
		2, 3, 5, 6,
		4, 5, 7, 8,
		5, 6, 8, 9,
	}
	expectedShape3D := []int{4, 4}

	result3D, err3D := input3D.im2col(kernelSize, stride)
	if err3D != nil {
		t.Fatalf("im2col (3D) returned error: %v", err3D)
	}
	expectedTensor3D := NewTensor(expectedData3D, expectedShape3D)
	if !tensorsAreEqual(expectedTensor3D, result3D, float32EqualityThreshold) {
		t.Errorf("im2col (3D) failed. Expected %v, got %v", expectedTensor3D, result3D)
	}

	input4D := NewTensor(input3D.Data, []int{1, 1, 3, 3})
	result4D, err4D := input4D.im2col(kernelSize, stride)
	if err4D != nil {
		t.Fatalf("im2col (4D) returned error: %v", err4D)
	}
	if !tensorsAreEqual(expectedTensor3D, result4D, float32EqualityThreshold) {
		t.Errorf("im2col (4D) failed. Expected %v, got %v", expectedTensor3D, result4D)
	}

	input2D := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	_, err2D := input2D.im2col(1, 1)
	if err2D == nil {
		t.Errorf("im2col with 2D input did not return error")
	}
}

func TestPad2D(t *testing.T) {
	input3D := NewTensor([]float32{1, 2, 3, 4}, []int{1, 2, 2})
	pad := 1
	expected3D := NewTensor([]float32{
		0, 0, 0, 0,
		0, 1, 2, 0,
		0, 3, 4, 0,
		0, 0, 0, 0,
	}, []int{1, 4, 4})

	result3D := input3D.Pad2D(pad)
	if !tensorsAreEqual(expected3D, result3D, float32EqualityThreshold) {
		t.Errorf("Pad2D (3D) failed. Expected %v, got %v", expected3D, result3D)
	}

	input4D := NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 1, 2, 2})
	expected4D := NewTensor([]float32{
		0, 0, 0, 0,
		0, 1, 2, 0,
		0, 3, 4, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 5, 6, 0,
		0, 7, 8, 0,
		0, 0, 0, 0,
	}, []int{2, 1, 4, 4})

	result4D := input4D.Pad2D(pad)
	if !tensorsAreEqual(expected4D, result4D, float32EqualityThreshold) {
		t.Errorf("Pad2D (4D) failed. Expected %v, got %v", expected4D, result4D)
	}

	resultPad0 := input3D.Pad2D(0)
	if !tensorsAreEqual(input3D, resultPad0, float32EqualityThreshold) {
		t.Errorf("Pad2D(0) should return a clone, but it differs.")
	}
	resultPad0.Data[0] = 99
	if input3D.Data[0] == 99 {
		t.Errorf("Pad2D(0) did not return a clone (modified original)")
	}

	input2D := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	checkPanic(t, func() { input2D.Pad2D(1) }, "only works for 3D or 4D")
}

func TestRepeat(t *testing.T) {
	input2D := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	expected2D_0 := NewTensor([]float32{1, 2, 3, 4, 1, 2, 3, 4}, []int{4, 2})
	result2D_0 := input2D.Repeat(0, 2)
	if !tensorsAreEqual(expected2D_0, result2D_0, float32EqualityThreshold) {
		t.Errorf("Repeat (2D, dim=0) failed. Expected %v, got %v", expected2D_0, result2D_0)
	}
	expected2D_1 := NewTensor([]float32{1, 2, 1, 2, 3, 4, 3, 4}, []int{2, 4})
	result2D_1 := input2D.Repeat(1, 2)
	if !tensorsAreEqual(expected2D_1, result2D_1, float32EqualityThreshold) {
		t.Errorf("Repeat (2D, dim=1) failed. Expected %v, got %v", expected2D_1, result2D_1)
	}

	input4D := NewTensor([]float32{1, 2, 3, 4}, []int{1, 1, 2, 2})
	expected4D_0 := NewTensor([]float32{1, 2, 3, 4, 1, 2, 3, 4}, []int{2, 1, 2, 2})
	result4D_0 := input4D.Repeat(0, 2)
	if !tensorsAreEqual(expected4D_0, result4D_0, float32EqualityThreshold) {
		t.Errorf("Repeat (4D, dim=0) failed. Expected %v, got %v", expected4D_0, result4D_0)
	}
	expected4D_1 := NewTensor([]float32{1, 2, 3, 4, 1, 2, 3, 4}, []int{1, 2, 2, 2})
	result4D_1 := input4D.Repeat(1, 2)
	if !tensorsAreEqual(expected4D_1, result4D_1, float32EqualityThreshold) {
		t.Errorf("Repeat (4D, dim=1) failed. Expected %v, got %v", expected4D_1, result4D_1)
	}

	checkPanic(t, func() { input2D.Repeat(2, 2) }, "Invalid dimension")
	checkPanic(t, func() { input4D.Repeat(2, 2) }, "Repeating along spatial dimensions")
	checkPanic(t, func() { input4D.Repeat(4, 2) }, "Invalid dimension")
	input3D := NewTensor(make([]float32, 4), []int{1, 2, 2})
	checkPanic(t, func() { input3D.Repeat(0, 2) }, "only supports 2D or 4D")
}

func TestPadAndCrop(t *testing.T) {
	input := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	padding := 1

	paddedExpected := NewTensor([]float32{
		0, 0, 0, 0,
		0, 1, 2, 0,
		0, 3, 4, 0,
		0, 0, 0, 0,
	}, []int{4, 4})
	paddedResult := input.Pad(padding)
	if !tensorsAreEqual(paddedExpected, paddedResult, float32EqualityThreshold) {
		t.Errorf("Pad failed. Expected %v, got %v", paddedExpected, paddedResult)
	}

	croppedResult := paddedResult.Crop(padding)
	if !tensorsAreEqual(input, croppedResult, float32EqualityThreshold) {
		t.Errorf("Crop failed. Expected %v, got %v", input, croppedResult)
	}

	crop0 := input.Crop(0)
	if crop0 != input {
		t.Errorf("Crop(0) should return the original tensor instance")
	}
}

func TestFlattenByDim(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("FlattenByDim fail unexpectedly.")
		}
	}()
	input := NewTensor(make([]float32, 24), []int{2, 3, 4})

	originalData := make([]float32, len(input.Data))
	copy(originalData, input.Data)

	tensorToFlatten1 := input.Clone()
	expectedShape1 := []int{2, 12}
	tensorToFlatten1.FlattenByDim(1, 2)
	if !reflect.DeepEqual(tensorToFlatten1.shape, expectedShape1) {
		t.Errorf("FlattenByDim(1, -1) failed. Expected shape %v, got %v", expectedShape1, tensorToFlatten1.shape)
	}
	if !floatSlicesAreEqual(originalData, tensorToFlatten1.Data, float32EqualityThreshold) {
		t.Errorf("FlattenByDim(1, -1) data was modified unexpectedly.")
	}

	tensorToFlatten2 := input.Clone()
	expectedShape2 := []int{6, 4}
	tensorToFlatten2.FlattenByDim(0, 1)
	if !reflect.DeepEqual(tensorToFlatten2.shape, expectedShape2) {
		t.Errorf("FlattenByDim(0, 1) failed. Expected shape %v, got %v", expectedShape2, tensorToFlatten2.shape)
	}
	if !floatSlicesAreEqual(originalData, tensorToFlatten2.Data, float32EqualityThreshold) {
		t.Errorf("FlattenByDim(0, 1) data was modified unexpectedly.")
	}

	tensorToFlatten3 := input.Clone()
	expectedShape3 := []int{24, 1}
	tensorToFlatten3.FlattenByDim(0, -1)
	if !reflect.DeepEqual(tensorToFlatten3.shape, expectedShape3) {
		t.Errorf("FlattenByDim(0, -1) failed. Expected shape %v, got %v", expectedShape3, tensorToFlatten3.shape)
	}
	if !floatSlicesAreEqual(originalData, tensorToFlatten3.Data, float32EqualityThreshold) {
		t.Errorf("FlattenByDim(0, -1) data was modified unexpectedly.")
	}

	tensorPanic := input.Clone()
	checkPanic(t, func() { tensorPanic.FlattenByDim(-1, 1) }, "Invalid startDim")
	checkPanic(t, func() { tensorPanic.FlattenByDim(0, 3) }, "Invalid endDim")
	checkPanic(t, func() { tensorPanic.FlattenByDim(2, 1) }, "startDim cannot be greater than endDim")
}

func TestGetCols(t *testing.T) {
	input := NewTensor([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, []int{2, 4})

	expected1_3 := NewTensor([]float32{2, 3, 6, 7}, []int{2, 2})
	result1_3 := input.GetCols(1, 3)
	if !tensorsAreEqual(expected1_3, result1_3, float32EqualityThreshold) {
		t.Errorf("GetCols(1, 3) failed. Expected %v, got %v", expected1_3, result1_3)
	}

	expected0_1 := NewTensor([]float32{1, 5}, []int{2, 1})
	result0_1 := input.GetCols(0, 1)
	if !tensorsAreEqual(expected0_1, result0_1, float32EqualityThreshold) {
		t.Errorf("GetCols(0, 1) failed. Expected %v, got %v", expected0_1, result0_1)
	}

	expected3_4 := NewTensor([]float32{4, 8}, []int{2, 1})
	result3_4 := input.GetCols(3, 4)
	if !tensorsAreEqual(expected3_4, result3_4, float32EqualityThreshold) {
		t.Errorf("GetCols(3, 4) failed. Expected %v, got %v", expected3_4, result3_4)
	}

	checkPanic(t, func() { input.GetCols(-1, 2) }, "Invalid column range")
	checkPanic(t, func() { input.GetCols(0, 5) }, "Invalid column range")
	checkPanic(t, func() { input.GetCols(2, 1) }, "Invalid column range")
	input3D := NewTensor(make([]float32, 8), []int{2, 2, 2})
	checkPanic(t, func() { input3D.GetCols(0, 1) }, "only works for 2D")
}

func TestSetCol(t *testing.T) {
	target := NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,
	}, []int{2, 3})
	colData := NewTensor([]float32{9, 8}, []int{2, 1})

	expected := NewTensor([]float32{
		1, 9, 3,
		4, 8, 6,
	}, []int{2, 3})

	target.SetCol(1, colData)
	if !tensorsAreEqual(expected, target, float32EqualityThreshold) {
		t.Errorf("SetCol failed. Expected %v, got %v", expected, target)
	}

	badDataShape1 := NewTensor([]float32{9, 8, 7}, []int{3, 1})
	colData1D := NewTensor([]float32{9, 8}, []int{2})

	targetReset := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	checkPanic(t, func() { targetReset.SetCol(1, badDataShape1) }, "Invalid column data dimensions")
	checkPanic(t, func() { targetReset.SetCol(1, colData1D) }, "Invalid column data dimensions")

	input3D := NewTensor(make([]float32, 8), []int{2, 2, 2})
	checkPanic(t, func() { input3D.SetCol(0, colData) }, "only works for 2D")
}

func TestGetCol(t *testing.T) {
	input := NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,
	}, []int{2, 3})

	expectedCol1 := NewTensor([]float32{2, 5}, []int{2})
	resultCol1 := input.GetCol(1)
	if !tensorsAreEqual(expectedCol1, resultCol1, float32EqualityThreshold) {
		t.Errorf("GetCol(1) failed. Expected %v, got %v", expectedCol1, resultCol1)
	}

	expectedCol0 := NewTensor([]float32{1, 4}, []int{2})
	resultCol0 := input.GetCol(0)
	if !tensorsAreEqual(expectedCol0, resultCol0, float32EqualityThreshold) {
		t.Errorf("GetCol(0) failed. Expected %v, got %v", expectedCol0, resultCol0)
	}

	checkPanic(t, func() { input.GetCol(-1) }, "Invalid column index")
	checkPanic(t, func() { input.GetCol(3) }, "Invalid column index")
	input3D := NewTensor(make([]float32, 8), []int{2, 2, 2})
	checkPanic(t, func() { input3D.GetCol(0) }, "only works for 2D")
}

func TestSumByDim(t *testing.T) {
	input := NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,
	}, []int{2, 3})

	expectedSum0 := NewTensor([]float32{5, 7, 9}, []int{3})
	resultSum0 := input.SumByDim(0)
	if !tensorsAreEqual(expectedSum0, resultSum0, float32EqualityThreshold) {
		t.Errorf("SumByDim(0) failed. Expected %v, got %v", expectedSum0, resultSum0)
	}

	expectedSum1 := NewTensor([]float32{6, 15}, []int{2})
	resultSum1 := input.SumByDim(1)
	if !tensorsAreEqual(expectedSum1, resultSum1, float32EqualityThreshold) {
		t.Errorf("SumByDim(1) failed. Expected %v, got %v", expectedSum1, resultSum1)
	}

	checkPanic(t, func() { input.SumByDim(2) }, "Invalid dimension")
	input3D := NewTensor(make([]float32, 8), []int{2, 2, 2})
	checkPanic(t, func() { input3D.SumByDim(0) }, "works for 2D tensors")
}

func TestMatMul(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	t2 := NewTensor([]float32{5, 6, 7, 8}, []int{2, 2})
	expected := NewTensor([]float32{
		1*5 + 2*7, 1*6 + 2*8,
		3*5 + 4*7, 3*6 + 4*8,
	}, []int{2, 2})

	result := t1.MatMul(t2)
	if !tensorsAreEqual(expected, result, float32EqualityThreshold) {
		t.Errorf("MatMul failed. Expected %v, got %v", expected, result)
	}

	t3 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	t4 := NewTensor([]float32{7, 8, 9, 10, 11, 12}, []int{3, 2})
	expected2 := NewTensor([]float32{
		1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12,
		4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12,
	}, []int{2, 2})
	result2 := t3.MatMul(t4)
	if !tensorsAreEqual(expected2, result2, float32EqualityThreshold) {
		t.Errorf("MatMul (non-square) failed. Expected %v, got %v", expected2, result2)
	}

	t_incompatible := NewTensor([]float32{1, 2}, []int{1, 2})
	t1.MatMul(t_incompatible)

	t1D := NewTensor([]float32{1, 2}, []int{2})
	t1.MatMul(t1D)

}

func TestExpand(t *testing.T) {
	row := NewTensor([]float32{1, 2, 3}, []int{1, 3})
	targetShapeRow := []int{4, 3}
	expectedRow := NewTensor([]float32{
		1, 2, 3,
		1, 2, 3,
		1, 2, 3,
		1, 2, 3,
	}, targetShapeRow)
	resultRow := row.Expand(targetShapeRow)
	if !tensorsAreEqual(expectedRow, resultRow, float32EqualityThreshold) {
		t.Errorf("Expand row failed. Expected %v, got %v", expectedRow, resultRow)
	}

	col := NewTensor([]float32{1, 2}, []int{2, 1})
	targetShapeCol := []int{2, 3}
	expectedCol := NewTensor([]float32{
		1, 1, 1,
		2, 2, 2,
	}, targetShapeCol)
	resultCol := col.Expand(targetShapeCol)
	if !tensorsAreEqual(expectedCol, resultCol, float32EqualityThreshold) {
		t.Errorf("Expand col failed. Expected %v, got %v", expectedCol, resultCol)
	}

	scalar := NewTensor([]float32{5}, []int{1, 1})
	targetShapeScalar := []int{2, 2}
	expectedScalar := NewTensor([]float32{5, 5, 5, 5}, targetShapeScalar)
	resultScalar := scalar.Expand(targetShapeScalar)
	if !tensorsAreEqual(expectedScalar, resultScalar, float32EqualityThreshold) {
		t.Errorf("Expand scalar failed. Expected %v, got %v", expectedScalar, resultScalar)
	}

	checkPanic(t, func() { row.Expand([]int{4, 3, 1}) }, "dimensions must match")
	checkPanic(t, func() { row.Expand([]int{4, 4}) }, "cannot expand dimension")
	nonUnit := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	checkPanic(t, func() { nonUnit.Expand([]int{2, 3}) }, "cannot expand dimension")
}

func TestGetRow(t *testing.T) {
	input := NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, []int{3, 3})

	expectedRow1 := NewTensor([]float32{4, 5, 6}, []int{1, 3})
	resultRow1 := input.GetRow(1)
	if !tensorsAreEqual(expectedRow1, resultRow1, float32EqualityThreshold) {
		t.Errorf("GetRow(1) failed. Expected %v, got %v", expectedRow1, resultRow1)
	}

	expectedRow0 := NewTensor([]float32{1, 2, 3}, []int{1, 3})
	resultRow0 := input.GetRow(0)
	if !tensorsAreEqual(expectedRow0, resultRow0, float32EqualityThreshold) {
		t.Errorf("GetRow(0) failed. Expected %v, got %v", expectedRow0, resultRow0)
	}

	checkPanic(t, func() { input.GetRow(-1) }, "row index out of range")
	checkPanic(t, func() { input.GetRow(3) }, "row index out of range")
	input1D := NewTensor([]float32{1, 2, 3}, []int{3})
	checkPanic(t, func() { input1D.GetRow(0) }, "requires 2D tensor")
}

func TestSigmoid(t *testing.T) {
	input := NewTensor([]float32{-1, 0, 1, 100, -100}, []int{5})
	expectedData := []float32{
		1.0 / (1.0 + math.Exp(1.0)),
		0.5,
		1.0 / (1.0 + math.Exp(-1.0)),
		1.0,
		0.0,
	}
	expected := NewTensor(expectedData, []int{5})

	result := input.Sigmoid()
	if !tensorsAreEqual(expected, result, float32EqualityThreshold) {
		t.Errorf("Sigmoid failed. Expected %v, got %v", expected, result)
	}

	input2D := NewTensor([]float32{0, 1, 2, 3}, []int{2, 2})
	result2D := input2D.Sigmoid()
	if !reflect.DeepEqual(input2D.shape, result2D.shape) {
		t.Errorf("Sigmoid did not preserve shape. Input %v, Output %v", input2D.shape, result2D.shape)
	}
}

func TestConv2D_Simple(t *testing.T) {
	input := NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, []int{1, 1, 3, 3})

	weights := NewTensor([]float32{
		1, 0,
		0, 0,
	}, []int{1, 1, 2, 2})

	kernelSize := 2
	stride := 1
	pad := 0

	expected := NewTensor([]float32{1, 2, 4, 5}, []int{1, 1, 2, 2})

	result, err := input.Conv2D(weights, kernelSize, stride, pad, pad)
	if err != nil {
		t.Fatalf("Conv2D failed with error: %v", err)
	}

	if !tensorsAreEqual(expected, result, float32EqualityThreshold) {
		t.Errorf("Conv2D (simple) failed.\nExpected: %v\nGot:      %v", expected, result)
	}

	input3D := NewTensor(input.Data, []int{1, 3, 3})
	result3D, err3D := input3D.Conv2D(weights, kernelSize, stride, pad, pad)
	if err3D != nil {
		t.Fatalf("Conv2D (3D input) failed with error: %v", err3D)
	}
	if !tensorsAreEqual(expected, result3D, float32EqualityThreshold) {
		t.Errorf("Conv2D (3D input) failed.\nExpected: %v\nGot:      %v", expected, result3D)
	}

	pad = 1
	inputPad := NewTensor([]float32{5}, []int{1, 1, 1, 1})
	weightsPad := NewTensor([]float32{1}, []int{1, 1, 1, 1})
	expectedPad := NewTensor([]float32{0, 0, 0, 0, 5, 0, 0, 0, 0}, []int{1, 1, 3, 3})
	resultPad, errPad := inputPad.Conv2D(weightsPad, 1, 1, pad, pad)
	if errPad != nil {
		t.Fatalf("Conv2D (padding) failed with error: %v", errPad)
	}
	if !tensorsAreEqual(expectedPad, resultPad, float32EqualityThreshold) {
		t.Errorf("Conv2D (padding) failed.\nExpected: %v\nGot:      %v", expectedPad, resultPad)
	}

	badWeights := NewTensor(make([]float32, 8), []int{1, 2, 2, 2})
	_, errMismatch := input.Conv2D(badWeights, kernelSize, stride, pad, pad)
	if errMismatch == nil || !contains(errMismatch.Error(), "weights shape mismatch") {
		t.Errorf("Conv2D did not return expected error for mismatched channels")
	}

	input2D := NewTensor([]float32{1, 2, 3, 4}, []int{2, 2})
	_, errInputDim := input2D.Conv2D(weights, kernelSize, stride, pad, pad)
	if errInputDim == nil || !contains(errInputDim.Error(), "must be 3D or 4D") {
		t.Errorf("Conv2D did not return expected error for 2D input")
	}
}
