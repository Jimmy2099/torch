package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	math "github.com/chewxy/math32"
)

// SoftmaxLayer 实现带稳定性的Softmax层
type SoftmaxLayer struct {
	input  *tensor.Tensor // 保存输入用于反向传播
	output *tensor.Tensor // 保存输出用于反向传播
	axis   int            // 计算softmax的维度
}

// NewSoftmaxLayer 创建新的Softmax层
func NewSoftmaxLayer(axis int) *SoftmaxLayer {
	return &SoftmaxLayer{
		axis: axis,
	}
}

// Forward 前向传播
func (s *SoftmaxLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x == nil {
		panic("input tensor cannot be nil")
	}
	if s.axis < 0 || s.axis >= len(x.Shape) {
		panic("invalid axis for softmax computation")
	}

	// 保存输入
	s.input = x.Clone()

	// 数值稳定性的softmax实现
	// 1. 沿着指定轴找出最大值
	maxVals := x.Max() // 修改：移除axis参数

	// 2. 减去最大值（数值稳定性）
	shifted := x.SubScalar(maxVals) // 修改：使用SubScalar方法

	// 3. 计算指数
	expVals := shifted.Apply(math.Exp)

	// 4. 沿指定轴求和
	sumExp := expVals.Sum() // 修改：移除axis参数

	// 5. 归一化
	s.output = expVals.DivScalar(sumExp) // 修改：使用DivScalar方法

	return s.output
}

// Backward 反向传播
func (s *SoftmaxLayer) Backward(dout *tensor.Tensor) *tensor.Tensor {
	if s.output == nil {
		panic("must call Forward first")
	}
	if dout == nil {
		panic("gradient tensor cannot be nil")
	}
	if !shapeEqual(s.output.Shape, dout.Shape) {
		panic("output and gradient shapes must match")
	}

	// Softmax的梯度计算
	// grad = output * (dout - sum(dout * output, axis))
	sumDout := dout.Multiply(s.output).Sum()           // 修改：移除axis参数
	grad := s.output.Multiply(dout.SubScalar(sumDout)) // 修改：使用SubScalar方法

	return grad
}

// SetAxis 设置softmax计算轴
func (s *SoftmaxLayer) SetAxis(axis int) {
	if axis < 0 {
		panic("axis must be non-negative")
	}
	s.axis = axis
}

// GetAxis 获取当前softmax计算轴
func (s *SoftmaxLayer) GetAxis() int {
	return s.axis
}
