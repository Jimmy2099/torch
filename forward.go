package torch

import "github.com/Jimmy2099/torch/data_struct/matrix"

// Forward performs forward propagation
func (nn *PolynomialRegressionNN) Forward(input *matrix.Matrix) *matrix.Matrix {
	nn.input = input

	// Hidden layer
	batchSize := input.Cols
	expandedBias1 := matrix.NewMatrix(10, batchSize)
	for i := 0; i < 10; i++ {
		for j := 0; j < batchSize; j++ {
			expandedBias1.Data[i][j] = nn.bias1.Data[i][0]
		}
	}

	nn.hiddenInput = matrix.Add(matrix.Multiply(nn.weights1, input), expandedBias1)
	nn.hiddenOutput = matrix.Apply(nn.hiddenInput, Relu)

	// Output layer
	expandedBias2 := matrix.NewMatrix(1, batchSize)
	for i := 0; i < 1; i++ {
		for j := 0; j < batchSize; j++ {
			expandedBias2.Data[i][j] = nn.bias2.Data[i][0]
		}
	}

	nn.outputInput = matrix.Add(matrix.Multiply(nn.weights2, nn.hiddenOutput), expandedBias2)
	nn.outputOutput = nn.outputInput // Linear output (no activation function)

	return nn.outputOutput
}
