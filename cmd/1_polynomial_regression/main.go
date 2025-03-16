package main

import (
	"fmt"
	"github.com/Jimmy2099/torch"
	"math"
	"math/rand"
	"time"

	"github.com/Jimmy2099/torch/data_struct/matrix"
)

// ReLULayer implements the ReLU activation function
type ReLULayer struct {
	Input     *matrix.Matrix
	Output    *matrix.Matrix
	GradInput *matrix.Matrix
}

func (l *ReLULayer) Parameters() []*matrix.Matrix {
	//TODO implement me
	panic("implement me")
}

// NewReLULayer creates a new ReLU layer
func NewReLULayer() *ReLULayer {
	return &ReLULayer{}
}

// Forward performs forward pass through the ReLU layer
func (l *ReLULayer) Forward(input *matrix.Matrix) *matrix.Matrix {
	l.Input = input
	l.Output = input.Apply(torch.Relu)
	return l.Output
}

// Backward performs backward pass through the ReLU layer
func (l *ReLULayer) Backward(gradOutput *matrix.Matrix, learningRate float64) *matrix.Matrix {
	l.GradInput = matrix.NewMatrix(l.Input.Rows, l.Input.Cols)

	// Element-wise multiplication with derivative of ReLU
	for i := 0; i < l.Input.Rows; i++ {
		for j := 0; j < l.Input.Cols; j++ {
			l.GradInput.Data[i][j] = gradOutput.Data[i][j] * torch.ReluDerivative(l.Input.Data[i][j])
		}
	}

	return l.GradInput
}

func (l *ReLULayer) ZeroGrad() {
	// Reset gradients
	l.GradInput = nil
}

// Neural Network implementation
type NeuralNetwork struct {
	Layers []torch.Layer
}

func (nn *NeuralNetwork) Parameters() []*matrix.Matrix {
	//TODO implement me
	panic("implement me")
}

// NewNeuralNetwork creates a new neural network with the specified layer dimensions
func NewNeuralNetwork(layerDims []int) *NeuralNetwork {
	nn := &NeuralNetwork{
		Layers: make([]torch.Layer, 0, len(layerDims)+len(layerDims)-2),
	}

	// Create layers
	for i := 0; i < len(layerDims)-1; i++ {
		nn.Layers = append(nn.Layers, torch.NewLinearLayer(layerDims[i], layerDims[i+1]))

		// Add ReLU activation except for the last layer
		if i < len(layerDims)-2 {
			nn.Layers = append(nn.Layers, NewReLULayer())
		}
	}

	return nn
}

// Forward performs forward pass through the neural network
func (nn *NeuralNetwork) Forward(input *matrix.Matrix) *matrix.Matrix {
	output := input
	for _, layer := range nn.Layers {
		output = layer.Forward(output)
	}
	return output
}

// Backward performs backward pass through the neural network
func (nn *NeuralNetwork) Backward(targets *matrix.Matrix, learningRate float64) {
	// Compute MSE loss gradient
	lastLayer := nn.Layers[len(nn.Layers)-1]
	output := lastLayer.(*torch.LinearLayer).Output

	// dL/dY = (Y - T) * 2/n
	gradOutput := matrix.Subtract(output, targets)
	batchSize := float64(targets.Cols)
	gradOutput = gradOutput.MulScalar(2.0 / batchSize)

	// Backprop through all layers
	for i := len(nn.Layers) - 1; i >= 0; i-- {
		gradOutput = nn.Layers[i].Backward(gradOutput, learningRate)
	}
}

// ZeroGrad resets all gradients in the neural network
func (nn *NeuralNetwork) ZeroGrad() {
	for _, layer := range nn.Layers {
		layer.ZeroGrad()
	}
}

// PolynomialFeatures generates polynomial features from the input
func polynomialFeatures(X *matrix.Matrix, degree int) *matrix.Matrix {
	return matrix.PolynomialFeatures(X, degree)
}

func targetFunc(x1, x2 float64) float64 {
	return 3 + 2*x1 + 1.5*x2 + 0.5*math.Pow(x1, 2) - 0.8*math.Pow(x2, 2) + 0.3*math.Pow(x1, 3)
}

func main() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Generate training data (10 samples, 2 features each)
	X_train := matrix.NewMatrix(2, 10)
	for i := 0; i < 2; i++ {
		for j := 0; j < 10; j++ {
			X_train.Data[i][j] = rand.Float64() //[0.0,1.0)
		}
	}

	// Define the target function
	y_train := matrix.NewMatrix(1, 10)
	for j := 0; j < 10; j++ {
		x1 := X_train.Data[0][j]
		x2 := X_train.Data[1][j]
		y_train.Data[0][j] = targetFunc(x1, x2)
	}

	// Generate polynomial features (degree=3)
	degree := 3
	X_train_poly := polynomialFeatures(X_train, degree)

	// Print dimensions
	fmt.Printf("X_train dimensions: (%d, %d)\n", X_train.Rows, X_train.Cols)
	fmt.Printf("X_train_poly dimensions: (%d, %d)\n", X_train_poly.Rows, X_train_poly.Cols)
	fmt.Printf("y_train dimensions: (%d, %d)\n", y_train.Rows, y_train.Cols)

	// Create neural network [input -> 10 -> 1]
	inputDim := X_train_poly.Rows
	hiddenDim := 10
	outputDim := 1
	model := NewNeuralNetwork([]int{inputDim, hiddenDim, outputDim})

	// Create trainer
	trainer := torch.NewBasicTrainer(torch.MSE)

	// Train model
	epochs := 500
	learningRate := 0.01
	trainer.Train(model, X_train_poly, y_train, epochs, learningRate)

	// Test model
	test_sample := matrix.NewMatrix(2, 1)

	test_sample.Data[0][0] = 0.2 //[0.0,1.0)
	test_sample.Data[1][0] = 0.3 //[0.0,1.0)

	test_sample_poly := polynomialFeatures(test_sample, degree)
	prediction := model.Forward(test_sample_poly)

	// Test model prediction  performance on data range [0.0,1.0)
	fmt.Printf("\nPredicted value: %.4f\n", prediction.Data[0][0])

	// Calculate true value
	trueValue := targetFunc(test_sample.Data[0][0], test_sample.Data[1][0])
	fmt.Printf("True value: %.4f\n", trueValue)

	// Print error
	fmt.Printf("Error: %.4f\n", math.Abs(prediction.Data[0][0]-trueValue))
}
