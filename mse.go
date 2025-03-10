package torch

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/matrix"
)

// MSE calculates Mean Squared Error between predictions and targets
func MSE(predictions, targets *matrix.Matrix) float64 {
	if predictions.Rows != targets.Rows || predictions.Cols != targets.Cols {
		panic(fmt.Sprintf("Matrix dimensions don't match for MSE: predictions(%d,%d), targets(%d,%d)",
			predictions.Rows, predictions.Cols, targets.Rows, targets.Cols))
	}

	sum := 0.0
	count := predictions.Rows * predictions.Cols

	for i := 0; i < predictions.Rows; i++ {
		for j := 0; j < predictions.Cols; j++ {
			diff := predictions.Data[i][j] - targets.Data[i][j]
			sum += diff * diff
		}
	}

	return sum / float64(count)
}
