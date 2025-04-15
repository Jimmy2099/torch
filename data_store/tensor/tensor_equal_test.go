package tensor

import "testing"

func TestTensorEqual(t *testing.T) {
	t.Run("Identical tensors", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0}
		t1 := NewTensor(data, []int{3})
		t2 := NewTensor(data, []int{3})
		if !t1.Equal(t2) {
			t.Error("Identical tensors should be equal")
		}
	})

	t.Run("Empty tensor comparison", func(t *testing.T) {
		empty1 := NewTensor([]float32{}, []int{0})
		empty2 := NewTensor([]float32{}, []int{0})
		if !empty1.Equal(empty2) {
			t.Error("Empty tensors should be equal")
		}
	})

	t.Run("Nil comparison", func(t *testing.T) {
		var nilTensor *Tensor
		nonNil := NewTensor(nil, []int{0})

		if nilTensor.Equal(nonNil) {
			t.Error("Nil and non-nil tensors should not be equal")
		}
	})
}
