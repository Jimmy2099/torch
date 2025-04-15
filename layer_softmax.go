package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	math "github.com/chewxy/math32"
)

type SoftmaxLayer struct {
	input  *tensor.Tensor // 保存输入用于反向传播
	output *tensor.Tensor // 保存输出用于反向传播
	axis   int            // 计算softmax的维度
}

func NewSoftmaxLayer(axis int) *SoftmaxLayer {
	return &SoftmaxLayer{
		axis: axis,
	}
}

func (s *SoftmaxLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x == nil {
		panic("input tensor cannot be nil")
	}
	if s.axis < 0 || s.axis >= len(x.Shape) {
		panic("invalid axis for softmax computation")
	}

	s.input = x.Clone()

	maxVals := x.Max() // 修改：移除axis参数

	shifted := x.SubScalar(maxVals) // 修改：使用SubScalar方法

	expVals := shifted.Apply(math.Exp)

	sumExp := expVals.Sum() // 修改：移除axis参数

	s.output = expVals.DivScalar(sumExp) // 修改：使用DivScalar方法

	return s.output
}

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

	sumDout := dout.Multiply(s.output).Sum()           // 修改：移除axis参数
	grad := s.output.Multiply(dout.SubScalar(sumDout)) // 修改：使用SubScalar方法

	return grad
}

func (s *SoftmaxLayer) SetAxis(axis int) {
	if axis < 0 {
		panic("axis must be non-negative")
	}
	s.axis = axis
}

func (s *SoftmaxLayer) GetAxis() int {
	return s.axis
}
