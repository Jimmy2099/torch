package torch

import "github.com/Jimmy2099/torch/data_struct/matrix"

// ReLULayer 激活层
type ReLULayer struct {
	input *matrix.Matrix // 保存输入用于反向传播
}

func NewReLULayer() *ReLULayer { return &ReLULayer{} }

func (r *ReLULayer) Forward(x *matrix.Matrix) *matrix.Matrix {
	r.input = x.Clone() // 保存输入
	return x.Apply(func(val float64) float64 {
		if val > 0 {
			return val
		}
		return 0
	})
}

func (r *ReLULayer) Backward(dout *matrix.Matrix) *matrix.Matrix {
	// 计算ReLU的梯度
	return r.input.Apply(func(val float64) float64 {
		if val > 0 {
			return 1
		}
		return 0
	}).Multiply(dout) // 与上游梯度相乘
}
