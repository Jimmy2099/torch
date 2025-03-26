package tensor

import "math"

// Copy 创建张量的深拷贝
func Copy(t *Tensor) *Tensor {
	data := make([]float64, len(t.Data))
	copy(data, t.Data)
	return NewTensor(data, append([]int{}, t.Shape...))
}

// Size 返回张量中元素的总数
func (t *Tensor) Size() int {
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// At 获取指定位置的元素（适用于任意维度）
func (t *Tensor) At(indices ...int) float64 {
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
	data := make([]float64, size)
	for i := range data {
		data[i] = 1.0
	}
	return &Tensor{Data: data, Shape: shape}
}

// Zeros 创建全0张量
func Zeros(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	return &Tensor{Data: data, Shape: shape}
}

// AddScalar 张量每个元素加标量
func (t *Tensor) AddScalar(scalar float64) *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = v + scalar
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Sub 张量减法
func (t *Tensor) Sub(other *Tensor) *Tensor {
	// 需要实现张量形状检查
	result := make([]float64, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] - other.Data[i]
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// MulScalar 张量每个元素乘标量
func (t *Tensor) MulScalar(scalar float64) *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = v * scalar
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Div 张量除法
func (t *Tensor) Div(other *Tensor) *Tensor {
	// 需要实现张量形状检查
	result := make([]float64, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] / other.Data[i]
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Sqrt 张量每个元素开平方
func (t *Tensor) Sqrt() *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = math.Sqrt(v)
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Pow 张量每个元素幂运算
func (t *Tensor) Pow(exponent float64) *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = math.Pow(v, exponent)
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Sum 沿指定维度求和
func (t *Tensor) SumByDim1(dims []int, keepDim bool) *Tensor {
	// 需要实现维度求和逻辑
	// 这里只是示例，实际需要更复杂的实现
	sum := 0.0
	for _, v := range t.Data {
		sum += v
	}
	return &Tensor{Data: []float64{sum}, Shape: []int{1}}
}

// Mul 张量乘法
func (t *Tensor) Mul(other *Tensor) *Tensor {
	// 需要实现张量形状检查
	result := make([]float64, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] * other.Data[i]
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Add 张量加法
func (t *Tensor) Add(other *Tensor) *Tensor {
	// 需要实现张量形状检查
	result := make([]float64, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] + other.Data[i]
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// ... existing code ...

// DivScalar 张量每个元素除标量
func (t *Tensor) DivScalar(scalar float64) *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = v / scalar
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Sum 不带参数的求和
func (t *Tensor) Sum111() *Tensor {
	sum := 0.0
	for _, v := range t.Data {
		sum += v
	}
	return &Tensor{Data: []float64{sum}, Shape: []int{1}}
}

// SumByDim1 沿指定维度求和
func (t *Tensor) SumByDim2(dims []int, keepDim bool) *Tensor {
	// 简化实现，实际应根据dims参数处理
	sum := 0.0
	for _, v := range t.Data {
		sum += v
	}
	return &Tensor{Data: []float64{sum}, Shape: []int{1}}
}

// Get 根据索引获取张量中的值
func (t *Tensor) Get(indices []int) float64 {
	idx := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}
	return t.Data[idx]
}

// Set 根据索引设置张量中的值
func (t *Tensor) Set1(indices []int, value float64) {
	idx := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}
	t.Data[idx] = value
}

// Max 返回张量中的最大值
func (t *Tensor) Max1() float64 {
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
	result := make([]float64, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] - other.Data[i]
	}
	return NewTensor(result, t.Shape)
}

// Sum 计算张量所有元素的和
func (t *Tensor) Sum1() float64 {
	sum := 0.0
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
	result := make([]float64, len(t.Data))
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
	result := make([]float64, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] * other.Data[i]
	}
	return NewTensor(result, t.Shape)
}

// Apply 对张量应用函数
func (t *Tensor) Apply1(f func(float64) float64) *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = f(v)
	}
	return NewTensor(result, t.Shape)
}

// Clone 克隆张量
func (t *Tensor) Clone1() *Tensor {
	data := make([]float64, len(t.Data))
	copy(data, t.Data)
	return NewTensor(data, t.Shape)
}

// shapeEqual 检查两个张量形状是否相同
func shapeEqual(shape1, shape2 []int) bool {
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
func (t *Tensor) SubScalar(scalar float64) *Tensor {
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = v - scalar
	}
	return NewTensor(result, t.Shape)
}
