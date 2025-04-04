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

// Reshape changes the shape of the tensor.  It does NOT allocate new memory.
func (t *Tensor) Reshape(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if size != t.Size() {
		panic("New shape is not compatible with existing size, " + fmt.Sprint("size:", size, " t.Size():", t.Size()))
	}

	t.Shape = shape
	return t
}

func (t *Tensor) Squeeze() *Tensor {
	newShape := make([]int, 0)
	for _, dim := range t.Shape {
		if dim != 1 {
			newShape = append(newShape, dim)
		}
	}
	return t.Reshape(newShape)
}

// 精确维度压缩方法
func (t *Tensor) SqueezeSpecific(dims []int) *Tensor {
	newShape := make([]int, 0)
	for i, dim := range t.Shape {
		shouldKeep := true
		for _, d := range dims {
			if i == d {
				if dim != 1 {
					panic(fmt.Sprintf("无法压缩维度 %d (size=%d != 1)", i, dim))
				}
				shouldKeep = false
				break
			}
		}
		if shouldKeep {
			newShape = append(newShape, dim)
		}
	}
	return t.Reshape(newShape)
}

// Indices 将线性索引转换为多维索引
func (t *Tensor) Indices(i int) []int {
	indices := make([]int, len(t.Shape))
	strides := computeStrides(t.Shape)
	for k := 0; k < len(t.Shape); k++ {
		if strides[k] == 0 {
			indices[k] = 0
			continue
		}
		indices[k] = i / strides[k]
		i = i % strides[k]
	}
	return indices
}
