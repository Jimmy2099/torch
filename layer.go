package torch

import (
	"github.com/Jimmy2099/torch/data_struct/tensor"
)

// Layer interface for neural network layers
type Layer interface {
	Forward(input *tensor.Tensor) *tensor.Tensor
	Backward(gradOutput *tensor.Tensor, learningRate float64) *tensor.Tensor
	ZeroGrad()
	Parameters() []*tensor.Tensor
}

type LayerLoader interface {
	SetWeightsAndShape(data []float64, shape []int)
	SetBiasAndShape(data []float64, shape []int)
}

type LayerForTesting interface {
	GetWeights() *tensor.Tensor
	GetBias() *tensor.Tensor
}

//Sequential forward container TODO

//// DenseLayer 全连接层接口
//type DenseLayer interface {
//	Layer
//	SetWeights(Weights *tensor.Tensor)
//	SetBias(bias *tensor.Tensor)
//}
//
//// ActivationLayer 激活函数层接口
//type ActivationLayer interface {
//	Layer
//}
//
//// DropoutLayer Dropout层接口
//type DropoutLayer interface {
//	Layer
//	SetDropoutRate(rate float64)
//}
//
//// BatchNormLayer 批归一化层接口
//type BatchNormLayer interface {
//	Layer
//	SetMomentum(momentum float64)
//}
//
//// ConvLayer 卷积层接口
//type ConvLayer interface {
//	Layer
//	SetKernel(kernel *tensor.Tensor)
//	SetStride(stride int)
//	SetPadding(padding int)
//}
//
//// PoolingLayer 池化层接口
//type PoolingLayer interface {
//	Layer
//	SetPoolSize(poolSize int)
//	SetStride(stride int)
//}
