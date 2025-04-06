package tensor

import (
	"fmt"
	math "github.com/chewxy/math32"
	"math/rand"
	"time"
)

// Copy 创建张量的深拷贝
func Copy(t *Tensor) *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	return NewTensor(data, append([]int{}, t.Shape...))
}

// Size 返回张量中元素的总数
func (t *Tensor) Size() int {
	return len(t.Data)
}

// At 获取指定位置的元素（适用于任意维度）
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

// Ones 创建全1张量
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

// Clamp 方法将每个元素限制在 [min, max] 范围内
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

// RandomNormal 标准正态分布
func RandomNormal(shape []int) *Tensor {
	// 注意：在实际应用中，建议在程序初始化时只调用一次 Seed，而不是每次调用函数时都 seed
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

// Random generates a tensor with random values in the range [min, max].
func Random(shape []int, min, max float32) *Tensor {
	if min > max {
		panic("min cannot be greater than max")
	}

	// Initialize the random seed only once (preferably outside this function in real applications)
	rand.Seed(time.Now().UnixNano())

	size := 1
	for _, dim := range shape {
		size *= dim
	}

	data := make([]float32, size)
	for i := range data {
		data[i] = min + float32(rand.Float32())*(max-min) // Scale to [min, max]
	}

	return &Tensor{Data: data, Shape: shape}
}

// Zeros 创建全0张量
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

// AddScalar 张量每个元素加标量
func (t *Tensor) AddScalar(scalar float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = v + scalar
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// MulScalar 张量每个元素乘标量
func (t *Tensor) MulScalar(scalar float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = v * scalar
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Sqrt 张量每个元素开平方
func (t *Tensor) Sqrt() *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = math.Sqrt(v)
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Pow 张量每个元素幂运算
func (t *Tensor) Pow(exponent float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = math.Pow(v, exponent)
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// 实现维度求和
func (t *Tensor) SumByDim1(dims []int, keepDims bool) *Tensor {
	// 验证输入维度
	for _, dim := range dims {
		if dim < 0 || dim >= len(t.Shape) {
			panic(fmt.Sprintf("invalid dimension %d for shape %v", dim, t.Shape))
		}
	}

	// 创建结果形状模板
	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)

	// 标记需要求和的维度
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

	// 初始化结果数据
	resultData := make([]float32, product(newShape))

	// 遍历原始数据
	for i := 0; i < len(t.Data); i++ {
		// 计算原始坐标
		originalIndices := make([]int, len(t.Shape))
		remainder := i
		for dim := len(t.Shape) - 1; dim >= 0; dim-- {
			originalIndices[dim] = remainder % t.Shape[dim]
			remainder /= t.Shape[dim]
		}

		// 计算结果坐标
		resultIndices := make([]int, len(newShape))
		for dim := 0; dim < len(newShape); dim++ {
			if reduceDims[dim] {
				resultIndices[dim] = 0
			} else {
				resultIndices[dim] = originalIndices[dim]
			}
		}

		// 转换为结果索引
		resultIndex := 0
		for dim := 0; dim < len(newShape); dim++ {
			resultIndex += resultIndices[dim] * resultStrides[dim]
		}

		// 累加值
		resultData[resultIndex] += t.Data[i]
	}

	// 处理是否保持维度
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

// DivScalar 张量每个元素除标量
func (t *Tensor) DivScalar(scalar float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = v / scalar
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Sum 不带参数的求和
func (t *Tensor) Sum111() *Tensor {
	var sum float32
	for _, v := range t.Data {
		sum += v
	}
	return &Tensor{Data: []float32{sum}, Shape: []int{1}}
}

// Get 根据索引获取张量中的值
func (t *Tensor) Get(indices []int) float32 {
	idx := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}
	return t.Data[idx]
}

// Set 根据索引设置张量中的值
func (t *Tensor) Set1(indices []int, value float32) {
	idx := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}
	t.Data[idx] = value
}

// Max 返回张量中的最大值
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

// Sub 张量减法
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

// Sum 计算张量所有元素的和
func (t *Tensor) Sum1() float32 {
	var sum float32
	for _, v := range t.Data {
		sum += v
	}
	return sum
}

// Div 张量除法
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

// Multiply 张量乘法
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

// Apply 对张量应用函数
func (t *Tensor) Apply1(f func(float32) float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = f(v)
	}
	return NewTensor(result, t.Shape)
}

// Clone 克隆张量
func (t *Tensor) Clone1() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	return NewTensor(data, t.Shape)
}

// shapeEqual 检查两个张量形状是否相同
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

// SubScalar 张量减去标量
func (t *Tensor) SubScalar(scalar float32) *Tensor {
	result := make([]float32, len(t.Data))
	for i, v := range t.Data {
		result[i] = v - scalar
	}
	return NewTensor(result, t.Shape)
}
