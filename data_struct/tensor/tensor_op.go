package tensor

import (
	"math/rand"
)

func NewTensorWithShape(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &Tensor{
		Data:  make([]float64, size),
		Shape: shape,
	}
}

func NewRandomTensor(shape []int) *Tensor {
	t := NewTensorWithShape(shape)
	for i := range t.Data {
		t.Data[i] = rand.Float64()*2 - 1 // Random between -1 and 1
	}
	return t
}

func NewTensorFromSlice(data [][]float64) *Tensor {
	rows := len(data)
	if rows == 0 {
		return NewTensorWithShape([]int{0, 0})
	}
	cols := len(data[0])

	flatData := make([]float64, 0, rows*cols)
	for i := 0; i < rows; i++ {
		if len(data[i]) != cols {
			panic("All rows must have the same length")
		}
		flatData = append(flatData, data[i]...)
	}
	return &Tensor{
		Data:  flatData,
		Shape: []int{rows, cols},
	}
}

// Reshape changes the shape of the tensor.  It does NOT allocate new memory.
func (t *Tensor) Reshape(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if size != t.Size() {
		panic("New shape is not compatible with existing size")
	}

	t.Shape = shape
	return t
}
