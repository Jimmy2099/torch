package torch

import (
	"github.com/Jimmy2099/torch/data_struct/matrix"
)

// MSE 均方误差损失函数
func MSE(predictions, targets *matrix.Matrix) float64 {
	diff := matrix.Subtract(predictions, targets)
	squared := matrix.Apply(diff, func(x float64) float64 { return x * x })
	return matrix.Sum(squared) / float64(squared.Rows)
}
