package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

func MSE(predictions, targets *tensor.Tensor) float32 {
	diff := predictions.Sub(targets)
	squared := diff.Apply(func(x float32) float32 { return x * x })
	return squared.Sum() / float32(squared.Shape[0])
}
