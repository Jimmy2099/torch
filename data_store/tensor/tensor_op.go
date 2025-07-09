package tensor

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	"math/rand"
)

func NewTensorWithShape(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &Tensor{
		Data:  make([]float32, size),
		shape: shape,
	}
}

func NewRandomTensor(shape []int) *Tensor {
	t := NewTensorWithShape(shape)
	for i := range t.Data {
		t.Data[i] = float32(rand.Float32())*2 - 1
	}
	return t
}

func NewTensorFromSlice(data [][]float32) *Tensor {
	rows := len(data)
	if rows == 0 {
		return NewTensorWithShape([]int{0, 0})
	}
	cols := len(data[0])

	flatData := make([]float32, 0, rows*cols)
	for i := 0; i < rows; i++ {
		if len(data[i]) != cols {
			panic("All rows must have the same length")
		}
		flatData = append(flatData, data[i]...)
	}
	return &Tensor{
		Data:  flatData,
		shape: []int{rows, cols},
	}
}

func (t *Tensor) Reshape(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if size != t.Size() {
		panic("New shape is not compatible with existing size, " + fmt.Sprint("size:", size, " t.Size():", t.Size()))
	}

	t.shape = shape
	return t
}

func (t *Tensor) Squeeze() *Tensor {
	newShape := make([]int, 0)
	for _, dim := range t.shape {
		if dim != 1 {
			newShape = append(newShape, dim)
		}
	}
	return t.Reshape(newShape)
}

func (t *Tensor) SqueezeSpecific(dims []int) *Tensor {
	newShape := make([]int, 0)
	for i, dim := range t.shape {
		shouldKeep := true
		for _, d := range dims {
			if i == d {
				if dim != 1 {
					panic(fmt.Sprintf("cannot squeeze dimension %d (size=%d != 1)", i, dim))
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

func (t *Tensor) Indices(i int) []int {
	indices := make([]int, len(t.shape))
	strides := computeStrides(t.shape)
	for k := 0; k < len(t.shape); k++ {
		if strides[k] == 0 {
			indices[k] = 0
			continue
		}
		indices[k] = i / strides[k]
		i = i % strides[k]
	}
	return indices
}

func (t *Tensor) Fill(value float32) {
	for i := range t.Data {
		t.Data[i] = value
	}
}

func (t *Tensor) Negate() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	for i := 0; i < len(data); i++ {
		data[i] = -data[i]
	}
	return NewTensor(data, t.shape)
}

func (t *Tensor) Permute(perm []int) *Tensor {
	// Validate permutation
	if len(perm) != len(t.shape) {
		panic(fmt.Sprintf("Permutation length %d doesn't match tensor dimensions %d", len(perm), len(t.shape)))
	}

	// Check for valid permutation indices
	seen := make(map[int]bool)
	for _, p := range perm {
		if p < 0 || p >= len(t.shape) {
			panic(fmt.Sprintf("Invalid permutation index: %d", p))
		}
		if seen[p] {
			panic("Duplicate permutation index")
		}
		seen[p] = true
	}

	// Calculate new shape
	newShape := make([]int, len(perm))
	for i, idx := range perm {
		newShape[i] = t.shape[idx]
	}

	// Handle zero-size tensor
	if len(t.Data) == 0 {
		return &Tensor{
			Data:  []float32{},
			shape: newShape,
		}
	}

	// Calculate strides for original tensor
	oldStrides := make([]int, len(t.shape))
	stride := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		oldStrides[i] = stride
		stride *= t.shape[i]
	}

	// Calculate strides for new tensor
	newStrides := make([]int, len(perm))
	stride = 1
	for i := len(perm) - 1; i >= 0; i-- {
		newStrides[i] = stride
		stride *= newShape[i]
	}

	// Create new data array
	totalSize := 1
	for _, s := range newShape {
		totalSize *= s
	}
	newData := make([]float32, totalSize)

	// Rearrange data according to permutation
	for i := 0; i < totalSize; i++ {
		origIndex := 0
		remainder := i
		for j := 0; j < len(perm); j++ {
			coord := remainder / newStrides[j]
			remainder = remainder % newStrides[j]
			origIndex += coord * oldStrides[perm[j]]
		}
		newData[i] = t.Data[origIndex]
	}

	return &Tensor{
		Data:  newData,
		shape: newShape,
	}
}

func (t *Tensor) Transpose() *Tensor {
	if len(t.shape) != 2 {
		panic("Transpose without permutation requires 2D tensor")
	}
	return t.Permute([]int{1, 0})
}
