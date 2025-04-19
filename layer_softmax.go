package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	math "github.com/chewxy/math32"
)

type SoftmaxLayer struct {
	input  *tensor.Tensor
	output *tensor.Tensor
	axis   int
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
	if s.axis < 0 || s.axis >= len(x.GetShape()) {
		panic("invalid axis for softmax computation")
	}

	s.input = x.Clone()

	maxVals := x.Max()

	shifted := x.SubScalar(maxVals)

	expVals := shifted.Apply(math.Exp)

	sumExp := expVals.Sum()

	s.output = expVals.DivScalar(sumExp)

	return s.output
}

func (s *SoftmaxLayer) Backward(dout *tensor.Tensor) *tensor.Tensor {
	if s.output == nil {
		panic("must call Forward first")
	}
	if dout == nil {
		panic("gradient tensor cannot be nil")
	}
	if !shapeEqual(s.output.GetShape(), dout.GetShape()) {
		panic("output and gradient shapes must match")
	}

	sumDout := dout.Multiply(s.output).Sum()
	grad := s.output.Multiply(dout.SubScalar(sumDout))

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
