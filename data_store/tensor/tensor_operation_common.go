package tensor

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"math/rand"
	"time"
)

func Copy(t *Tensor) *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	return NewTensor(data, append([]int{}, t.Shape...))
}

func (t *Tensor) Size() int {
	return len(t.Data)
}

func (t *Tensor) At(indices ...int) float32 {
	if len(indices) != len(t.Shape) {
		panic("number of indices must match tensor rank")
	}

	pos := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.Shape[i] {
			panic("tensor: index out of range")
		}
		pos += indices[i] * stride
		stride *= t.Shape[i]
	}

	return t.Data[pos]
}

func Ones(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = 1.0
	}
	return &Tensor{Data: data, Shape: shape}
}

func (t *Tensor) Clamp(min, max float32) *Tensor {
	clampedData := make([]float32, len(t.Data))
	for i, val := range t.Data {
		if val < min {
			clampedData[i] = min
		} else if val > max {
			clampedData[i] = max
		} else {
			clampedData[i] = float32(val)
		}
	}
	return &Tensor{Data: clampedData, Shape: t.Shape}
}

func RandomNormal(shape []int) *Tensor {
	rand.Seed(time.Now().UnixNano())

	size := 1
	for _, dim := range shape {
		size *= dim
	}

	data := make([]float32, size)
	for i := range data {
		data[i] = float32(rand.NormFloat64()) // 标准正态分布：均值0，标准差1
	}

	return &Tensor{Data: data, Shape: shape}
}

func Random(shape []int, min, max float32) *Tensor {
	if min > max {
		panic("min cannot be greater than max")
	}

	rand.Seed(time.Now().UnixNano())

	size := 1
	for _, dim := range shape {
		size *= dim
	}

	data := make([]float32, size)
	for i := range data {
		data[i] = min + rand.Float32()*(max-min) // Scale to [min, max]
	}

	return &Tensor{Data: data, Shape: shape}
}

func Zeros(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float32, size)
	return &Tensor{Data: data, Shape: shape}
}

func ZerosLike(t *Tensor) *Tensor {
	if t == nil {
		panic("tensor.ZerosLike: input tensor cannot be nil")
	}

	if t.Shape == nil {
		panic(fmt.Sprintf("tensor.ZerosLike: input tensor (value: %p) has a nil shape", t))
	}
	shape := t.Shape
	zeroTensor := Zeros(shape)
	return zeroTensor
}

func (t *Tensor) AddScalar(scalar float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = v + scalar
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

func (t *Tensor) MulScalar(scalar float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = v * scalar
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

func (t *Tensor) Sqrt() *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = math.Sqrt(v)
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

func (t *Tensor) Pow(exponent float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = math.Pow(v, exponent)
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

func (t *Tensor) SumByDim1(dims []int, keepDims bool) *Tensor {
	for _, dim := range dims {
		if dim < 0 || dim >= len(t.Shape) {
			panic(fmt.Sprintf("invalid dimension %d for shape %v", dim, t.Shape))
		}
	}

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)

	reduceDims := make(map[int]bool)
	for _, dim := range dims {
		reduceDims[dim] = true
		newShape[dim] = 1
	}

	resultStrides := make([]int, len(newShape))
	stride := 1
	for i := len(newShape) - 1; i >= 0; i-- {
		resultStrides[i] = stride
		stride *= newShape[i]
	}

	resultData := make([]float32, product(newShape))

	for i := 0; i < len(t.Data); i++ {
		originalIndices := make([]int, len(t.Shape))
		remainder := i
		for dim := len(t.Shape) - 1; dim >= 0; dim-- {
			originalIndices[dim] = remainder % t.Shape[dim]
			remainder /= t.Shape[dim]
		}

		resultIndices := make([]int, len(newShape))
		for dim := 0; dim < len(newShape); dim++ {
			if reduceDims[dim] {
				resultIndices[dim] = 0
			} else {
				resultIndices[dim] = originalIndices[dim]
			}
		}

		resultIndex := 0
		for dim := 0; dim < len(newShape); dim++ {
			resultIndex += resultIndices[dim] * resultStrides[dim]
		}

		resultData[resultIndex] += t.Data[i]
	}

	if !keepDims {
		finalShape := make([]int, 0)
		for dim := 0; dim < len(newShape); dim++ {
			if !reduceDims[dim] {
				finalShape = append(finalShape, newShape[dim])
			}
		}
		newShape = finalShape
	}

	return &Tensor{
		Data:  resultData,
		Shape: newShape,
	}
}

func (t *Tensor) DivScalar(scalar float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = v / scalar
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

func (t *Tensor) Sum111() *Tensor {
	var sum float32
	for _, v := range t.Data {
		sum += v
	}
	return &Tensor{Data: []float32{sum}, Shape: []int{1}}
}

func (t *Tensor) Get(indices []int) float32 {
	idx := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}
	return t.Data[idx]
}

func (t *Tensor) Set1(indices []int, value float32) {
	idx := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}
	t.Data[idx] = value
}

func (t *Tensor) Max1() float32 {
	if len(t.Data) == 0 {
		return 0
	}
	max := t.Data[0]
	for _, v := range t.Data {
		if v > max {
			max = v
		}
	}
	return max
}

func (t *Tensor) Sub1(other *Tensor) *Tensor {
	if !shapeEqual(t.Shape, other.Shape) {
		panic("shape mismatch in tensor subtraction")
	}
	result := make([]float32, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] - other.Data[i]
	}
	return NewTensor(result, t.Shape)
}

func (t *Tensor) Sum1() float32 {
	var sum float32
	for _, v := range t.Data {
		sum += v
	}
	return sum
}

func (t *Tensor) Div1(other *Tensor) *Tensor {
	if !shapeEqual(t.Shape, other.Shape) {
		panic("shape mismatch in tensor division")
	}
	result := make([]float32, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] / other.Data[i]
	}
	return NewTensor(result, t.Shape)
}

func (t *Tensor) Multiply1(other *Tensor) *Tensor {
	if !shapeEqual(t.Shape, other.Shape) {
		panic("shape mismatch in tensor multiplication")
	}
	result := make([]float32, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] * other.Data[i]
	}
	return NewTensor(result, t.Shape)
}

func (t *Tensor) Apply1(f func(float32) float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = f(v)
	}
	return NewTensor(result, t.Shape)
}

func (t *Tensor) Clone1() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	return NewTensor(data, t.Shape)
}

func ShapeEqual(shape1, shape2 []int) bool {
	if len(shape1) != len(shape2) {
		return false
	}
	for i := range shape1 {
		if shape1[i] != shape2[i] {
			return false
		}
	}
	return true
}

func (t *Tensor) SubScalar(scalar float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = v - scalar
	}
	return NewTensor(result, t.Shape)
}
