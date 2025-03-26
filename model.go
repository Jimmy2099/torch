package torch

import "github.com/Jimmy2099/torch/data_struct/tensor"

// Model 定义模型接口
type Model interface {
	Forward(input *tensor.Tensor) *tensor.Tensor
	Backward(target *tensor.Tensor, learningRate float64)
	Parameters() []*tensor.Tensor
	ZeroGrad()
}

// Trainer 定义训练器接口
type Trainer interface {
	Train(model Model, inputs, targets *tensor.Tensor, epochs int, learningRate float64)
	Validate(model Model, inputs, targets *tensor.Tensor) float64
}
