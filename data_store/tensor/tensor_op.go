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
	{
		inferPos := -1
		inferPosCount := 0
		for i := 0; i < len(shape); i++ {
			if shape[i] == -1 {
				inferPos = i
				inferPosCount += 1
			}
		}

		if t.Size() != 0 && inferPosCount > 0 {
			if inferPosCount > 1 {
				panic("inferPosCount should be less than 2")
			}
			total := 1
			for i := 0; i < len(shape); i++ {
				if shape[i] < 0 {
					continue
				}
				total *= shape[i]
			}
			unKnowPosVal := t.Size() / total
			shape[inferPos] = unKnowPosVal
		}
	}

	newSize := 1
	if len(shape) > 0 {
		for _, dim := range shape {
			newSize *= dim
		}
	} else if len(shape) == 0 {
		newSize = 1
	}
	if newSize != t.Size() {
		panic("New shape is not compatible with existing size, " + fmt.Sprint("size:", newSize, " t.Size():", t.Size()))
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
	if len(perm) != len(t.shape) {
		panic(fmt.Sprintf("Permutation length %d doesn't match tensor dimensions %d", len(perm), len(t.shape)))
	}

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

	newShape := make([]int, len(perm))
	for i, idx := range perm {
		newShape[i] = t.shape[idx]
	}

	if len(t.Data) == 0 {
		return &Tensor{
			Data:  []float32{},
			shape: newShape,
		}
	}

	oldStrides := make([]int, len(t.shape))
	stride := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		oldStrides[i] = stride
		stride *= t.shape[i]
	}

	newStrides := make([]int, len(perm))
	stride = 1
	for i := len(perm) - 1; i >= 0; i-- {
		newStrides[i] = stride
		stride *= newShape[i]
	}

	totalSize := 1
	for _, s := range newShape {
		totalSize *= s
	}
	newData := make([]float32, totalSize)

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

func (t *Tensor) ScatterAdd(indices *Tensor, source *Tensor) {
	if len(indices.shape) != 1 {
		panic("ScatterAdd indices must be 1D")
	}

	if indices.shape[0] != source.shape[0] {
		panic("ScatterAdd indices and source must have same first dimension")
	}

	if len(t.shape) != len(source.shape) {
		panic("ScatterAdd target and source must have same number of dimensions")
	}

	for i := 1; i < len(t.shape); i++ {
		if t.shape[i] != source.shape[i] {
			panic("ScatterAdd target and source must have same dimensions after the first")
		}
	}

	rowSize := ShapeSum(t.shape[1:])

	for i := 0; i < indices.shape[0]; i++ {
		idx := int(indices.Data[i])
		if idx < 0 || idx >= t.shape[0] {
			panic(fmt.Sprintf("ScatterAdd index out of range: %d not in [0, %d)", idx, t.shape[0]))
		}

		for j := 0; j < rowSize; j++ {
			t.Data[idx*rowSize+j] += source.Data[i*rowSize+j]
		}
	}
}

func (t *Tensor) Gather(indices *Tensor, axis int) *Tensor {
	inputRank := t.Rank()
	if axis < 0 {
		axis += inputRank
	}

	if axis < 0 || axis >= inputRank {
		panic(fmt.Errorf("invalid axis %d for tensor of rank %d", axis, inputRank))
	}

	outputShape := make([]int, 0, inputRank-1+indices.Rank())
	outputShape = append(outputShape, t.shape[:axis]...)
	outputShape = append(outputShape, indices.shape...)
	outputShape = append(outputShape, t.shape[axis+1:]...)

	M := shapeSum(t.shape[:axis])
	N := indices.Size()
	blockSize := shapeSum(t.shape[axis+1:])

	inputBatchStride := shapeSum(t.shape[axis:])
	outputBatchStride := N * blockSize

	axisDimLimit := int(t.shape[axis])

	outputData := make([]float32, shapeSum(outputShape))

	for m := 0; m < M; m++ {
		srcBatchOffset := m * inputBatchStride
		dstBatchOffset := m * outputBatchStride

		for i := 0; i < N; i++ {
			idx := int(indices.Data[i])

			if idx < 0 {
				idx += axisDimLimit
			}

			if idx < 0 || idx >= axisDimLimit {
				panic(fmt.Errorf("gather index out of range: %d is not in [0, %d)", idx, axisDimLimit))
			}

			srcOffset := srcBatchOffset + idx*blockSize
			dstOffset := dstBatchOffset + i*blockSize

			copy(outputData[dstOffset:dstOffset+blockSize], t.Data[srcOffset:srcOffset+blockSize])
		}
	}

	return NewTensor(outputData, outputShape)
}
