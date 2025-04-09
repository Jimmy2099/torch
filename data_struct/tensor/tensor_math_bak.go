package tensor

import "github.com/Jimmy2099/torch/pkg/log"

// Sub 张量减法
func (t *Tensor) Sub_bak(other *Tensor) *Tensor {
	defer func() {
		if err := recover(); err != nil {
			log.Println("t.Shape:", t.Shape, " other Shape:", other.Shape)
			panic(err)
		}
	}()
	// 需要实现张量形状检查
	result := make([]float32, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] - other.Data[i]
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Div 张量除法
func (t *Tensor) Div_bak(other *Tensor) *Tensor {
	// 需要实现张量形状检查
	result := make([]float32, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] / other.Data[i]
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Mul 张量乘法
func (t *Tensor) Mul_bak(other *Tensor) *Tensor {
	// 需要实现张量形状检查
	result := make([]float32, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] * other.Data[i]
	}
	return &Tensor{Data: result, Shape: t.Shape}
}

// Add 张量加法
func (t *Tensor) Add_bak(other *Tensor) *Tensor {
	// 需要实现张量形状检查
	result := make([]float32, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] + other.Data[i]
	}
	return &Tensor{Data: result, Shape: t.Shape}
}
