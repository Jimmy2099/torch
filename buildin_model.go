package torch

import "github.com/Jimmy2099/torch/data_struct/matrix"

// PolynomialRegressionNN represents a neural network for polynomial regression
type PolynomialRegressionNN struct {
	inputDim int

	weights1 *matrix.Matrix // Weights from input to hidden
	bias1    *matrix.Matrix // Bias for hidden layer
	weights2 *matrix.Matrix // Weights from hidden to output
	bias2    *matrix.Matrix // Bias for output layer

	// Cached values for backpropagation
	input        *matrix.Matrix
	hiddenInput  *matrix.Matrix
	hiddenOutput *matrix.Matrix
	outputInput  *matrix.Matrix
	outputOutput *matrix.Matrix
}
