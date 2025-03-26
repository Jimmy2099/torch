package torch

import (
	"github.com/Jimmy2099/torch/data_struct/tensor"
)

// MSE 均方误差损失函数
func MSE(predictions, targets *tensor.Tensor) float64 {
	diff := predictions.Sub(targets)                                // 使用Tensor的Sub方法
	squared := diff.Apply(func(x float64) float64 { return x * x }) // 使用Tensor的Apply方法
	return squared.Sum() / float64(squared.Shape[0])                // 使用Tensor的Sum方法和Shape属性
}
