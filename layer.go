package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Layer interface {
	Forward(input *tensor.Tensor) *tensor.Tensor

	//DEL
	Backward(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor
	ZeroGrad()
	Parameters() []*tensor.Tensor
	//DEL
}

type LayerLoader interface {
	SetWeightsAndShape(data []float32, shape []int)
	SetBiasAndShape(data []float32, shape []int)
}

type LayerForTesting interface {
	GetWeights() *tensor.Tensor
	GetBias() *tensor.Tensor
}
