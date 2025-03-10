package torch

import "github.com/Jimmy2099/torch/data_struct/matrix"

// Model 定义模型接口
type Model interface {
	Forward(input *matrix.Matrix) *matrix.Matrix
	Backward(target *matrix.Matrix, learningRate float64)
	Parameters() []*matrix.Matrix
	ZeroGrad()
}

// Layer interface for neural network layers
type Layer interface {
	Forward(input *matrix.Matrix) *matrix.Matrix
	Backward(gradOutput *matrix.Matrix, learningRate float64) *matrix.Matrix
	ZeroGrad()
}

// Trainer 定义训练器接口
type Trainer interface {
	Train(model Model, inputs, targets *matrix.Matrix, epochs int, learningRate float64)
	Validate(model Model, inputs, targets *matrix.Matrix) float64
}
