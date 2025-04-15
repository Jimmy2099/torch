package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	math "github.com/chewxy/math32"
)

type SigmoidLayer struct {
}

func NewSigmoidLayer() *SigmoidLayer {
	return &SigmoidLayer{}
}

func Sigmoid(x float32) float32 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidDerivative(x float32) float32 {
	s := Sigmoid(x)
	return s * (1.0 - s)
}

func (s *SigmoidLayer) Forward(input *tensor.Tensor) *tensor.Tensor {
	shape := input.Shape

	outputData := make([]float32, len(input.Data))
	for i := 0; i < len(input.Data); i++ {
		outputData[i] = Sigmoid(input.Data[i])
	}

	return tensor.NewTensor(outputData, shape)
}

func (s *SigmoidLayer) Backward(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor {
	shape := gradOutput.Shape

	gradInputData := make([]float32, len(gradOutput.Data))
	for i := 0; i < len(gradOutput.Data); i++ {
		gradInputData[i] = gradOutput.Data[i] * SigmoidDerivative(gradOutput.Data[i])
	}

	return tensor.NewTensor(gradInputData, shape)
}

func (s *SigmoidLayer) ZeroGrad() {
}

func (s *SigmoidLayer) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}
