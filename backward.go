package torch

import "github.com/Jimmy2099/torch/data_struct/matrix"

// Backward performs backpropagation
func (nn *PolynomialRegressionNN) Backward(target *matrix.Matrix, learningRate float64) {
	batchSize := nn.input.Cols

	// Output layer error
	outputError := matrix.Subtract(nn.outputOutput, target)

	// Gradients for output layer (no activation derivative for linear output)
	outputGradient := outputError

	weightsOutputDelta := matrix.Multiply(outputGradient, matrix.Transpose(nn.hiddenOutput))
	biasOutputDelta := matrix.NewMatrix(1, 1)
	for i := 0; i < 1; i++ {
		sum := 0.0
		for j := 0; j < batchSize; j++ {
			sum += outputGradient.Data[i][j]
		}
		biasOutputDelta.Data[i][0] = sum / float64(batchSize)
	}

	// Hidden layer error
	hiddenError := matrix.Multiply(matrix.Transpose(nn.weights2), outputGradient)

	// Gradients for hidden layer
	hiddenGradient := matrix.Apply(nn.hiddenInput, ReluDerivative)
	hiddenGradient = matrix.HadamardProduct(hiddenGradient, hiddenError)

	weightsHiddenDelta := matrix.Multiply(hiddenGradient, matrix.Transpose(nn.input))
	biasHiddenDelta := matrix.NewMatrix(10, 1)
	for i := 0; i < 10; i++ {
		sum := 0.0
		for j := 0; j < batchSize; j++ {
			sum += hiddenGradient.Data[i][j]
		}
		biasHiddenDelta.Data[i][0] = sum / float64(batchSize)
	}

	// Update weights and biases
	for i := 0; i < nn.weights2.Rows; i++ {
		for j := 0; j < nn.weights2.Cols; j++ {
			nn.weights2.Data[i][j] -= learningRate * weightsOutputDelta.Data[i][j] / float64(batchSize)
		}
		nn.bias2.Data[i][0] -= learningRate * biasOutputDelta.Data[i][0]
	}

	for i := 0; i < nn.weights1.Rows; i++ {
		for j := 0; j < nn.weights1.Cols; j++ {
			nn.weights1.Data[i][j] -= learningRate * weightsHiddenDelta.Data[i][j] / float64(batchSize)
		}
		nn.bias1.Data[i][0] -= learningRate * biasHiddenDelta.Data[i][0]
	}
}
