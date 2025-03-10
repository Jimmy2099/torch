package torch

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/matrix"
)

// Train trains the neural network
func (nn *PolynomialRegressionNN) Train(inputs, targets *matrix.Matrix, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		// Forward pass
		outputs := nn.Forward(inputs)

		// Calculate loss
		loss := MSE(outputs, targets)

		// Backward pass
		nn.Backward(targets, learningRate)

		if (epoch+1)%50 == 0 {
			fmt.Printf("Epoch [%d/%d], Loss: %.4f\n", epoch+1, epochs, loss)
		}
	}
}
