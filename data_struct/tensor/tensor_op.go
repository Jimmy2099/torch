package tensor

import (
	"fmt"
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

func (t *Tensor) Reshape(newShape []int) {
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	if newSize != len(t.Data) {
		panic(fmt.Sprintf("New shape %v is incompatible with data size %d", newShape, len(t.Data)))
	}
	t.Shape = newShape
}
