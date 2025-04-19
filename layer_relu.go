package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ReLULayer struct {
	input    *tensor.Tensor
	negative float32
	inplace  bool
}

func (r *ReLULayer) GetWeights() *tensor.Tensor {
	return tensor.NewEmptyTensor()
}

func (r *ReLULayer) GetBias() *tensor.Tensor {
	return tensor.NewEmptyTensor()
}

func NewReLULayer() *ReLULayer {
	return &ReLULayer{
		negative: 0.0,
		inplace:  false,
	}
}

func NewLeakyReLULayer(negativeSlope float32) *ReLULayer {
	if negativeSlope < 0 {
		panic("negative slope must be non-negative")
	}
	return &ReLULayer{
		negative: negativeSlope,
		inplace:  false,
	}
}

func (r *ReLULayer) SetInplace(inplace bool) {
	r.inplace = inplace
}

func (r *ReLULayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x == nil {
		panic("input tensor cannot be nil")
	}

	if r.inplace {
		r.input = x
	} else {
		r.input = x.Clone()
	}

	return x.Apply(func(val float32) float32 {
		if val > 0 {
			return val
		}
		return r.negative * val
	})
}

func shapeEqual(a, b []int) bool {
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

func (r *ReLULayer) ActivationType() string {
	if r.negative == 0 {
		return "ReLU"
	}
	return "LeakyReLU"
}

func (r *ReLULayer) NegativeSlope() float32 {
	return r.negative
}

func (r *ReLULayer) Backward(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor {
	if r.input == nil {
		panic("must call Forward first")
	}
	if gradOutput == nil {
		panic("gradient tensor cannot be nil")
	}
	if !shapeEqual(r.input.GetShape(), gradOutput.GetShape()) {
		panic("input and gradient shapes must match")
	}

	grad := r.input.Apply(func(val float32) float32 {
		if val > 0 {
			return 1.0
		}
		return r.negative
	})

	return grad.Multiply(gradOutput)
}

func (r *ReLULayer) ZeroGrad() {
}

func (r *ReLULayer) Parameters() []*tensor.Tensor {
	return nil
}
