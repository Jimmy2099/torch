package tensor

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
)

func (t *Tensor) Flatten() *Tensor {
	return NewTensor(t.Data, []int{len(t.Data)})
}

func (t *Tensor) Set(value float32, indices ...int) {
	pos := 0
	stride := 1
	for i := len(indices) - 1; i >= 0; i-- {
		pos += indices[i] * stride
		if i > 0 {
			stride *= t.shape[i]
		}
	}
	t.Data[pos] = value
}

func Multiply(a, b *Tensor) *Tensor {
	if len(a.shape) < 2 || len(b.shape) < 2 {
		panic("Tensors must have at least 2 dimensions for multiplication")
	}

	if a.shape[len(a.shape)-1] != b.shape[len(b.shape)-2] {
		panic(fmt.Sprintf("Tensor dimensions don't match for multiplication: %v * %v",
			a.shape, b.shape))
	}

	if len(a.shape) > 2 && len(b.shape) > 2 {
		if !equal(a.shape[:len(a.shape)-2], b.shape[:len(b.shape)-2]) {
			panic(fmt.Sprintf("Leading tensor dimensions don't match: %v vs %v",
				a.shape[:len(a.shape)-2], b.shape[:len(b.shape)-2]))
		}
	}

	outShape := make([]int, max(len(a.shape), len(b.shape)))
	copy(outShape, a.shape)
	outShape[len(outShape)-1] = b.shape[len(b.shape)-1]

	aRows := a.shape[len(a.shape)-2]
	aCols := a.shape[len(a.shape)-1]
	bCols := b.shape[len(b.shape)-1]

	totalElements := 1
	for i := 0; i < len(outShape); i++ {
		totalElements *= outShape[i]
	}

	resultData := make([]float32, totalElements)

	indices := make([]int, len(outShape)-2)
	for {
		aOffset := 0
		bOffset := 0
		stride := 1
		for i := len(indices) - 1; i >= 0; i-- {
			aOffset += indices[i] * stride
			bOffset += indices[i] * stride
			stride *= a.shape[i]
		}

		for i := 0; i < aRows; i++ {
			for j := 0; j < bCols; j++ {
				var sum float32
				for k := 0; k < aCols; k++ {
					aPos := aOffset + i*aCols + k
					bPos := bOffset + k*bCols + j
					sum += a.Data[aPos] * b.Data[bPos]
				}
				resultPos := 0
				stride := 1
				for d := len(outShape) - 1; d >= 0; d-- {
					if d == len(outShape)-2 {
						resultPos += i * stride
					} else if d == len(outShape)-1 {
						resultPos += j * stride
					} else {
						resultPos += indices[d] * stride
					}
					if d > 0 {
						stride *= outShape[d]
					}
				}
				resultData[resultPos] = sum
			}
		}

		if !incrementIndices(indices, a.shape[:len(a.shape)-2]) {
			break
		}
	}

	return NewTensor(resultData, outShape)
}

func Add(a, b *Tensor) *Tensor {
	if !equal(a.shape, b.shape) {
		panic(fmt.Sprintf("Tensor shapes don't match for addition: %v + %v", a.shape, b.shape))
	}

	resultData := make([]float32, len(a.Data))
	for i := range a.Data {
		resultData[i] = a.Data[i] + b.Data[i]
	}

	return NewTensor(resultData, a.shape)
}

func Subtract(a, b *Tensor) *Tensor {
	if !equal(a.shape, b.shape) {
		panic(fmt.Sprintf("Tensor shapes don't match for subtraction: %v - %v", a.shape, b.shape))
	}

	resultData := make([]float32, len(a.Data))
	for i := range a.Data {
		resultData[i] = a.Data[i] - b.Data[i]
	}

	return NewTensor(resultData, a.shape)
}

func HadamardProduct(a, b *Tensor) *Tensor {
	if !equal(a.shape, b.shape) {
		panic(fmt.Sprintf("Tensor shapes don't match for Hadamard product: %v * %v", a.shape, b.shape))
	}

	resultData := make([]float32, len(a.Data))
	for i := range a.Data {
		resultData[i] = a.Data[i] * b.Data[i]
	}

	return NewTensor(resultData, a.shape)
}

func Transpose(t *Tensor, dims ...int) *Tensor {
	if len(dims) == 0 {
		if len(t.shape) < 2 {
			return t.Copy()
		}
		dims = make([]int, len(t.shape))
		for i := range dims {
			dims[i] = i
		}
		dims[len(dims)-1], dims[len(dims)-2] = dims[len(dims)-2], dims[len(dims)-1]
	}

	if len(dims) != len(t.shape) {
		panic(fmt.Sprintf("Invalid transpose dimensions: got %d, expected %d", len(dims), len(t.shape)))
	}

	newShape := make([]int, len(t.shape))
	for i, dim := range dims {
		newShape[i] = t.shape[dim]
	}

	resultData := make([]float32, len(t.Data))

	oldIndices := make([]int, len(t.shape))
	for {
		oldPos := 0
		stride := 1
		for i := len(oldIndices) - 1; i >= 0; i-- {
			oldPos += oldIndices[i] * stride
			if i > 0 {
				stride *= t.shape[i]
			}
		}

		newIndices := make([]int, len(dims))
		for i, dim := range dims {
			newIndices[i] = oldIndices[dim]
		}

		newPos := 0
		stride = 1
		for i := len(newIndices) - 1; i >= 0; i-- {
			newPos += newIndices[i] * stride
			if i > 0 {
				stride *= newShape[i]
			}
		}

		resultData[newPos] = t.Data[oldPos]

		if !incrementIndices(oldIndices, t.shape) {
			break
		}
	}

	return NewTensor(resultData, newShape)
}

func (t *Tensor) Apply(fn func(float32) float32) *Tensor {
	resultData := make([]float32, len(t.Data))
	for i, val := range t.Data {
		resultData[i] = fn(val)
	}
	return NewTensor(resultData, t.shape)
}

func (t *Tensor) Sum() float32 {
	var sum float32
	for _, val := range t.Data {
		sum += float32(val)
	}
	return sum
}

func (t *Tensor) Mean() float32 {
	return t.Sum() / float32(len(t.Data))
}

func (t *Tensor) Max() float32 {
	maxVal := math.Inf(-1)
	for _, val := range t.Data {
		if val > maxVal {
			maxVal = float32(val)
		}
	}
	return maxVal
}

func (t *Tensor) Min() float32 {
	maxVal := math.Inf(1)
	for _, val := range t.Data {
		if val < maxVal {
			maxVal = float32(val)
		}
	}
	return maxVal
}

func (t *Tensor) Copy() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	return NewTensor(data, t.shape)
}

func (t *Tensor) ReLU() *Tensor {
	return t.Apply(func(x float32) float32 {
		return math.Max(0, x)
	})
}

func (t *Tensor) Softmax() *Tensor {
	if len(t.shape) < 1 {
		panic("Tensor must have at least 1 dimension for softmax")
	}

	lastDim := t.shape[len(t.shape)-1]
	otherDims := 1
	for i := 0; i < len(t.shape)-1; i++ {
		otherDims *= t.shape[i]
	}

	result := NewTensor(make([]float32, len(t.Data)), t.shape)

	for i := 0; i < otherDims; i++ {
		maxVal := math.Inf(-1)
		for j := 0; j < lastDim; j++ {
			val := t.Data[i*lastDim+j]
			if val > maxVal {
				maxVal = float32(val)
			}
		}

		var sum float32
		exps := make([]float32, lastDim)
		for j := 0; j < lastDim; j++ {
			exps[j] = math.Exp(t.Data[i*lastDim+j] - maxVal)
			sum += exps[j]
		}

		for j := 0; j < lastDim; j++ {
			result.Data[i*lastDim+j] = exps[j] / sum
		}
	}

	return result
}

func (t *Tensor) ArgMax() *Tensor {
	if len(t.shape) < 1 {
		panic("Tensor must have at least 1 dimension for argmax")
	}

	lastDim := t.shape[len(t.shape)-1]
	otherDims := 1
	for i := 0; i < len(t.shape)-1; i++ {
		otherDims *= t.shape[i]
	}

	resultShape := make([]int, len(t.shape)-1)
	copy(resultShape, t.shape[:len(t.shape)-1])
	result := NewTensor(make([]float32, otherDims), resultShape)

	for i := 0; i < otherDims; i++ {
		maxIdx := 0
		maxVal := t.Data[i*lastDim]
		for j := 1; j < lastDim; j++ {
			if t.Data[i*lastDim+j] > maxVal {
				maxVal = t.Data[i*lastDim+j]
				maxIdx = j
			}
		}
		result.Data[i] = float32(maxIdx)
	}

	return result
}

func (t *Tensor) MaxPool(poolSize, stride int) (*Tensor, *Tensor) {
	if len(t.shape) < 2 {
		panic("Tensor must have at least 2 dimensions for max pooling")
	}

	rows := t.shape[len(t.shape)-2]
	cols := t.shape[len(t.shape)-1]
	outRows := (rows-poolSize)/stride + 1
	outCols := (cols-poolSize)/stride + 1

	outShape := make([]int, len(t.shape))
	copy(outShape, t.shape)
	outShape[len(outShape)-2] = outRows
	outShape[len(outShape)-1] = outCols

	totalElements := 1
	for _, dim := range outShape {
		totalElements *= dim
	}

	resultData := make([]float32, totalElements)
	argmaxData := make([]float32, totalElements)

	indices := make([]int, len(t.shape)-2)
	for {
		offset := 0
		stride := 1
		for i := len(indices) - 1; i >= 0; i-- {
			offset += indices[i] * stride
			stride *= t.shape[i]
		}

		for i := 0; i < outRows; i++ {
			for j := 0; j < outCols; j++ {
				maxVal := math.Inf(-1)
				maxIdx := 0

				for pi := 0; pi < poolSize; pi++ {
					for pj := 0; pj < poolSize; pj++ {
						row := i*stride + pi
						col := j*stride + pj
						if row < rows && col < cols {
							pos := offset + row*cols + col
							if t.Data[pos] > maxVal {
								maxVal = t.Data[pos]
								maxIdx = pos
							}
						}
					}
				}

				resultPos := 0
				stride := 1
				for d := len(outShape) - 1; d >= 0; d-- {
					if d == len(outShape)-2 {
						resultPos += i * stride
					} else if d == len(outShape)-1 {
						resultPos += j * stride
					} else {
						resultPos += indices[d] * stride
					}
					if d > 0 {
						stride *= outShape[d]
					}
				}

				resultData[resultPos] = maxVal
				argmaxData[resultPos] = float32(maxIdx)
			}
		}

		if !incrementIndices(indices, t.shape[:len(t.shape)-2]) {
			break
		}
	}

	return NewTensor(resultData, outShape), NewTensor(argmaxData, outShape)
}

func equal(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func incrementIndices(indices, shape []int) bool {
	for i := len(indices) - 1; i >= 0; i-- {
		indices[i]++
		if indices[i] < shape[i] {
			return true
		}
		indices[i] = 0
	}
	return false
}

func (t *Tensor) ShapesMatch(other *Tensor) bool {
	if t == nil || other == nil {
		return false
	}
	if len(t.shape) != len(other.shape) {
		return false
	}

	for i := range t.shape {
		if t.shape[i] != other.shape[i] {
			return false
		}
	}

	return true
}
