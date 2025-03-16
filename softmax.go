package torch

import (
	"github.com/Jimmy2099/torch/data_struct/matrix"
	"math"
)

func Softmax(m *matrix.Matrix) *matrix.Matrix {
	res := matrix.NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		maxVal := -math.MaxFloat64
		for j := 0; j < m.Cols; j++ {
			if m.Data[i][j] > maxVal {
				maxVal = m.Data[i][j]
			}
		}

		sumExp := 0.0
		for j := 0; j < m.Cols; j++ {
			res.Data[i][j] = math.Exp(m.Data[i][j] - maxVal)
			sumExp += res.Data[i][j]
		}

		for j := 0; j < m.Cols; j++ {
			res.Data[i][j] /= sumExp
		}
	}
	return res
}
