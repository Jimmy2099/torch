package tensor

import (
	"fmt"
	math "github.com/chewxy/math32"
)

// Flatten returns a 1D tensor with all elements
func (t *Tensor) Flatten() *Tensor {
	return NewTensor(t.Data, []int{len(t.Data)})
}

// Set sets the element at the given indices
func (t *Tensor) Set(value float32, indices ...int) {
	pos := 0
	stride := 1
	for i := len(indices) - 1; i >= 0; i-- {
		pos += indices[i] * stride
		if i > 0 {
			stride *= t.Shape[i]
		}
	}
	t.Data[pos] = value
}

// Multiply performs tensor multiplication (dot product) on the last two dimensions
func Multiply(a, b *Tensor) *Tensor {
	if len(a.Shape) < 2 || len(b.Shape) < 2 {
		panic("Tensors must have at least 2 dimensions for multiplication")
	}

	// Check if the last dimension of a matches the second last dimension of b
	if a.Shape[len(a.Shape)-1] != b.Shape[len(b.Shape)-2] {
		panic(fmt.Sprintf("Tensor dimensions don't match for multiplication: %v * %v",
			a.Shape, b.Shape))
	}

	// For higher dimensions, we need to check if the leading dimensions match
	if len(a.Shape) > 2 && len(b.Shape) > 2 {
		if !equal(a.Shape[:len(a.Shape)-2], b.Shape[:len(b.Shape)-2]) {
			panic(fmt.Sprintf("Leading tensor dimensions don't match: %v vs %v",
				a.Shape[:len(a.Shape)-2], b.Shape[:len(b.Shape)-2]))
		}
	}

	// Calculate output shape
	outShape := make([]int, max(len(a.Shape), len(b.Shape)))
	copy(outShape, a.Shape)
	outShape[len(outShape)-1] = b.Shape[len(b.Shape)-1]

	// Flatten the last two dimensions for easier computation
	aRows := a.Shape[len(a.Shape)-2]
	aCols := a.Shape[len(a.Shape)-1]
	bCols := b.Shape[len(b.Shape)-1]

	totalElements := 1
	for i := 0; i < len(outShape); i++ {
		totalElements *= outShape[i]
	}

	resultData := make([]float32, totalElements)

	// Iterate through all possible indices except the last two
	indices := make([]int, len(outShape)-2)
	for {
		// Calculate offsets for a and b
		aOffset := 0
		bOffset := 0
		stride := 1
		for i := len(indices) - 1; i >= 0; i-- {
			aOffset += indices[i] * stride
			bOffset += indices[i] * stride
			stride *= a.Shape[i]
		}

		// Perform matrix multiplication on the last two dimensions
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

		// Increment indices
		if !incrementIndices(indices, a.Shape[:len(a.Shape)-2]) {
			break
		}
	}

	return NewTensor(resultData, outShape)
}

// Add performs element-wise addition
func Add(a, b *Tensor) *Tensor {
	if !equal(a.Shape, b.Shape) {
		panic(fmt.Sprintf("Tensor shapes don't match for addition: %v + %v", a.Shape, b.Shape))
	}

	resultData := make([]float32, len(a.Data))
	for i := range a.Data {
		resultData[i] = a.Data[i] + b.Data[i]
	}

	return NewTensor(resultData, a.Shape)
}

// Subtract performs element-wise subtraction
func Subtract(a, b *Tensor) *Tensor {
	if !equal(a.Shape, b.Shape) {
		panic(fmt.Sprintf("Tensor shapes don't match for subtraction: %v - %v", a.Shape, b.Shape))
	}

	resultData := make([]float32, len(a.Data))
	for i := range a.Data {
		resultData[i] = a.Data[i] - b.Data[i]
	}

	return NewTensor(resultData, a.Shape)
}

// HadamardProduct performs element-wise multiplication
func HadamardProduct(a, b *Tensor) *Tensor {
	if !equal(a.Shape, b.Shape) {
		panic(fmt.Sprintf("Tensor shapes don't match for Hadamard product: %v * %v", a.Shape, b.Shape))
	}

	resultData := make([]float32, len(a.Data))
	for i := range a.Data {
		resultData[i] = a.Data[i] * b.Data[i]
	}

	return NewTensor(resultData, a.Shape)
}

// Transpose returns the transpose of the tensor
func Transpose(t *Tensor, dims ...int) *Tensor {
	if len(dims) == 0 {
		// Default to reversing the last two dimensions
		if len(t.Shape) < 2 {
			return t.Copy()
		}
		dims = make([]int, len(t.Shape))
		for i := range dims {
			dims[i] = i
		}
		// Swap last two dimensions
		dims[len(dims)-1], dims[len(dims)-2] = dims[len(dims)-2], dims[len(dims)-1]
	}

	// Validate permutation
	if len(dims) != len(t.Shape) {
		panic(fmt.Sprintf("Invalid transpose dimensions: got %d, expected %d", len(dims), len(t.Shape)))
	}

	// Calculate new shape
	newShape := make([]int, len(t.Shape))
	for i, dim := range dims {
		newShape[i] = t.Shape[dim]
	}

	// Create mapping from old indices to new indices
	resultData := make([]float32, len(t.Data))

	oldIndices := make([]int, len(t.Shape))
	for {
		// Calculate position in original tensor
		oldPos := 0
		stride := 1
		for i := len(oldIndices) - 1; i >= 0; i-- {
			oldPos += oldIndices[i] * stride
			if i > 0 {
				stride *= t.Shape[i]
			}
		}

		// Calculate new indices
		newIndices := make([]int, len(dims))
		for i, dim := range dims {
			newIndices[i] = oldIndices[dim]
		}

		// Calculate position in new tensor
		newPos := 0
		stride = 1
		for i := len(newIndices) - 1; i >= 0; i-- {
			newPos += newIndices[i] * stride
			if i > 0 {
				stride *= newShape[i]
			}
		}

		resultData[newPos] = t.Data[oldPos]

		// Increment indices
		if !incrementIndices(oldIndices, t.Shape) {
			break
		}
	}

	return NewTensor(resultData, newShape)
}

// Apply applies a function to each element of the tensor
func (t *Tensor) Apply(fn func(float32) float32) *Tensor {
	resultData := make([]float32, len(t.Data))
	for i, val := range t.Data {
		resultData[i] = fn(val)
	}
	return NewTensor(resultData, t.Shape)
}

// Sum returns the sum of all elements in the tensor
func (t *Tensor) Sum() float32 {
	var sum float32
	for _, val := range t.Data {
		sum += float32(val)
	}
	return sum
}

// Mean returns the mean of all elements in the tensor
func (t *Tensor) Mean() float32 {
	return t.Sum() / float32(len(t.Data))
}

// Max returns the maximum value in the tensor
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

// Copy returns a deep copy of the tensor
func (t *Tensor) Copy() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	return NewTensor(data, t.Shape)
}

// ReLU applies the rectified linear unit activation function
func (t *Tensor) ReLU() *Tensor {
	return t.Apply(func(x float32) float32 {
		return math.Max(0, x)
	})
}

// Softmax applies the softmax function along the last dimension
func (t *Tensor) Softmax() *Tensor {
	if len(t.Shape) < 1 {
		panic("Tensor must have at least 1 dimension for softmax")
	}

	// Reshape to 2D tensor where each row is a set of values to softmax
	lastDim := t.Shape[len(t.Shape)-1]
	otherDims := 1
	for i := 0; i < len(t.Shape)-1; i++ {
		otherDims *= t.Shape[i]
	}

	// Create result tensor
	result := NewTensor(make([]float32, len(t.Data)), t.Shape)

	// Apply softmax to each row
	for i := 0; i < otherDims; i++ {
		// Find max value for numerical stability
		maxVal := math.Inf(-1)
		for j := 0; j < lastDim; j++ {
			val := t.Data[i*lastDim+j]
			if val > maxVal {
				maxVal = float32(val)
			}
		}

		// Compute exponentials and sum
		var sum float32
		exps := make([]float32, lastDim)
		for j := 0; j < lastDim; j++ {
			exps[j] = math.Exp(t.Data[i*lastDim+j] - maxVal)
			sum += exps[j]
		}

		// Compute softmax values
		for j := 0; j < lastDim; j++ {
			result.Data[i*lastDim+j] = exps[j] / sum
		}
	}

	return result
}

// ArgMax returns the indices of the maximum values along the last dimension
func (t *Tensor) ArgMax() *Tensor {
	if len(t.Shape) < 1 {
		panic("Tensor must have at least 1 dimension for argmax")
	}

	lastDim := t.Shape[len(t.Shape)-1]
	otherDims := 1
	for i := 0; i < len(t.Shape)-1; i++ {
		otherDims *= t.Shape[i]
	}

	// Create result tensor with one less dimension
	resultShape := make([]int, len(t.Shape)-1)
	copy(resultShape, t.Shape[:len(t.Shape)-1])
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

// MaxPool performs max pooling on the last two dimensions
func (t *Tensor) MaxPool(poolSize, stride int) (*Tensor, *Tensor) {
	if len(t.Shape) < 2 {
		panic("Tensor must have at least 2 dimensions for max pooling")
	}

	// Calculate output dimensions
	rows := t.Shape[len(t.Shape)-2]
	cols := t.Shape[len(t.Shape)-1]
	outRows := (rows-poolSize)/stride + 1
	outCols := (cols-poolSize)/stride + 1

	// Create output shape
	outShape := make([]int, len(t.Shape))
	copy(outShape, t.Shape)
	outShape[len(outShape)-2] = outRows
	outShape[len(outShape)-1] = outCols

	// Calculate total elements in output
	totalElements := 1
	for _, dim := range outShape {
		totalElements *= dim
	}

	// Create result and argmax tensors
	resultData := make([]float32, totalElements)
	argmaxData := make([]float32, totalElements)

	// Iterate through all dimensions except the last two
	indices := make([]int, len(t.Shape)-2)
	for {
		// Calculate offset for current indices
		offset := 0
		stride := 1
		for i := len(indices) - 1; i >= 0; i-- {
			offset += indices[i] * stride
			stride *= t.Shape[i]
		}

		// Perform max pooling on the last two dimensions
		for i := 0; i < outRows; i++ {
			for j := 0; j < outCols; j++ {
				// Find max in pool window
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

				// Calculate position in result tensor
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

		// Increment indices
		if !incrementIndices(indices, t.Shape[:len(t.Shape)-2]) {
			break
		}
	}

	return NewTensor(resultData, outShape), NewTensor(argmaxData, outShape)
}

// helper functions

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

// ShapesMatch checks if the shape of this tensor matches the shape of another tensor.
func (t *Tensor) ShapesMatch(other *Tensor) bool {
	// Check if either tensor is nil
	if t == nil || other == nil {
		return false // Or handle as appropriate, maybe panic?
	}
	// Check if the number of dimensions is the same
	if len(t.Shape) != len(other.Shape) {
		return false
	}

	// Check if each dimension size matches
	for i := range t.Shape {
		if t.Shape[i] != other.Shape[i] {
			return false
		}
	}

	// If all checks pass, the shapes match
	return true
}
