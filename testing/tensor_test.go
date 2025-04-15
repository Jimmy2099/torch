package testing

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"testing"
)

func TestGetTensorTestResult(t *testing.T) {
	// Shared base test tensor
	baseTensor := tensor.Ones([]int{2, 2})

	t.Run("Element-wise multiplication", func(t *testing.T) {
		script := `out = in1 * in2`
		result := GetTensorTestResult(script, baseTensor, baseTensor)
		expected := baseTensor.Mul(baseTensor)
		if !result.Equal(expected) {
			t.Errorf("Element-wise multiplication failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

	t.Run("Tensor addition", func(t *testing.T) {
		script := `out = in1 + in2`
		result := GetTensorTestResult(script, baseTensor, baseTensor)
		expected := baseTensor.Add(baseTensor)
		if !result.Equal(expected) {
			t.Errorf("Addition failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

	t.Run("Matrix multiplication", func(t *testing.T) {
		in1 := tensor.Ones([]int{2, 3})
		in2 := tensor.Ones([]int{3, 2})
		script := `out = in1 @ in2`
		result := GetTensorTestResult(script, in1, in2)
		expectedData := []float32{3, 3, 3, 3}
		expected := tensor.NewTensor(expectedData, []int{2, 2})
		if !result.Equal(expected) {
			t.Errorf("Matrix multiplication failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

	t.Run("Broadcasting", func(t *testing.T) {
		scalar := tensor.NewTensor([]float32{2}, []int{1})
		script := `out = in1 * in2`
		result := GetTensorTestResult(script, baseTensor, scalar)
		expectedData := []float32{2, 2, 2, 2}
		expected := tensor.NewTensor(expectedData, []int{2, 2})
		if !result.Equal(expected) {
			t.Errorf("Broadcast multiplication failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

	t.Run("Invalid matrix multiplication", func(t *testing.T) {
		in1 := tensor.Ones([]int{2, 3})
		in2 := tensor.Ones([]int{2, 3})
		script := `out = in1 @ in2`

		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic but none occurred")
			}
		}()
		GetTensorTestResult(script, in1, in2)
	})

	t.Run("Negative number operations", func(t *testing.T) {
		in1 := tensor.NewTensor([]float32{-1, 2, -3, 4}, []int{2, 2})
		in2 := tensor.NewTensor([]float32{2, 3, 4, -5}, []int{2, 2})
		script := `out = in1 * in2`
		result := GetTensorTestResult(script, in1, in2)
		expectedData := []float32{-2, 6, -12, -20}
		expected := tensor.NewTensor(expectedData, []int{2, 2})
		if !result.Equal(expected) {
			t.Errorf("Negative number multiplication failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

	t.Run("Invalid script", func(t *testing.T) {
		script := `out = in1 + undefined_var`
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic but none occurred")
			}
		}()
		GetTensorTestResult(script, baseTensor, baseTensor)
	})

	t.Run("High-dimensional broadcasting", func(t *testing.T) {
		in1 := tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{3, 1, 2})
		in2 := tensor.NewTensor([]float32{10, 20}, []int{2})
		script := `out = in1 + in2`
		expectedData := []float32{11, 22, 13, 24, 15, 26}
		expected := tensor.NewTensor(expectedData, []int{3, 1, 2})
		result := GetTensorTestResult(script, in1, in2)
		if !result.Equal(expected) {
			t.Errorf("High-dimensional broadcasting failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})
}
