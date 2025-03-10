package torch

import "github.com/Jimmy2099/torch/data_struct/matrix"

// Predict makes a prediction for input data
func (nn *PolynomialRegressionNN) Predict(input *matrix.Matrix) *matrix.Matrix {
	return nn.Forward(input)
}
