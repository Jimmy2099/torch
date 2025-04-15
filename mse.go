package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

// MSE 均方误差损失函数
func MSE(predictions, targets *tensor.Tensor) float32 {
	diff := predictions.Sub(targets)                                // 使用Tensor的Sub方法
	squared := diff.Apply(func(x float32) float32 { return x * x }) // 使用Tensor的Apply方法
	return squared.Sum() / float32(squared.Shape[0])                // 使用Tensor的Sum方法和Shape属性
}
