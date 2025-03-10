package torch

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/matrix"
	"math"
	"math/rand"
	"time"
)

func Xmain() {
	// Set random seed
	rand.Seed(time.Now().UnixNano())

	// Generate training data (10 samples, each with 2 features)
	numSamples := 10
	xTrain := matrix.NewMatrix(2, numSamples)
	yTrain := matrix.NewMatrix(1, numSamples)

	// Fill with random values between 0 and 1
	for i := 0; i < numSamples; i++ {
		xTrain.Data[0][i] = rand.Float64()
		xTrain.Data[1][i] = rand.Float64()
	}

	// Define true polynomial function:
	// y = 3 + 2*x1 + 1.5*x2 + 0.5*x1^2 - 0.8*x2^2 + 0.3*x1^3
	for i := 0; i < numSamples; i++ {
		x1 := xTrain.Data[0][i]
		x2 := xTrain.Data[1][i]
		yTrain.Data[0][i] = 3 + 2*x1 + 1.5*x2 + 0.5*math.Pow(x1, 2) - 0.8*math.Pow(x2, 2) + 0.3*math.Pow(x1, 3)
	}

	// Generate polynomial features up to degree 3
	degree := 3
	xTrainPoly := matrix.PolynomialFeatures(xTrain, degree)

	fmt.Printf("Original feature dimensions: (%d, %d)\n", xTrain.Rows, xTrain.Cols)
	fmt.Printf("Polynomial feature dimensions: (%d, %d)\n", xTrainPoly.Rows, xTrainPoly.Cols)

	// Create model
	inputDim := xTrainPoly.Rows
	model := NewPolynomialRegressionNN(inputDim)

	// Train model
	numEpochs := 500
	learningRate := 0.01
	model.Train(xTrainPoly, yTrain, numEpochs, learningRate)

	// Print learned parameters
	model.PrintParameters()

	// Prediction example
	testSample := matrix.NewMatrix(2, 1)
	testSample.Data[0][0] = 10
	testSample.Data[1][0] = 0.8

	testSamplePoly := matrix.PolynomialFeatures(testSample, degree)
	prediction := model.Predict(testSamplePoly)

	fmt.Printf("\nPredicted value: %.6f\n", prediction.Data[0][0])

	// Calculate true value
	x1 := testSample.Data[0][0]
	x2 := testSample.Data[1][0]
	trueValue := 3 + 2*x1 + 1.5*x2 + 0.5*math.Pow(x1, 2) - 0.8*math.Pow(x2, 2) + 0.3*math.Pow(x1, 3)
	fmt.Printf("True value: %.6f\n", trueValue)
}

// NewPolynomialRegressionNN creates a new polynomial regression neural network
func NewPolynomialRegressionNN(inputDim int) *PolynomialRegressionNN {
	return &PolynomialRegressionNN{
		inputDim: inputDim,
		weights1: matrix.RandomizeMatrix(10, inputDim), // Hidden layer with 10 neurons
		bias1:    matrix.RandomizeMatrix(10, 1),
		weights2: matrix.RandomizeMatrix(1, 10), // Output layer with 1 neuron
		bias2:    matrix.RandomizeMatrix(1, 1),
	}
}

// PrintParameters prints the model parameters
func (nn *PolynomialRegressionNN) PrintParameters() {
	fmt.Println("\nLearned Parameters:")
	fmt.Println("fc1.weight (first few rows):")
	for i := 0; i < min(3, nn.weights1.Rows); i++ {
		fmt.Printf("  ")
		for j := 0; j < nn.weights1.Cols; j++ {
			fmt.Printf("%.4f ", nn.weights1.Data[i][j])
		}
		fmt.Println()
	}

	fmt.Println("fc1.bias (first few values):")
	fmt.Printf("  ")
	for i := 0; i < min(5, nn.bias1.Rows); i++ {
		fmt.Printf("%.4f ", nn.bias1.Data[i][0])
	}
	fmt.Println()

	fmt.Println("fc2.weight:")
	fmt.Printf("  ")
	for j := 0; j < min(5, nn.weights2.Cols); j++ {
		fmt.Printf("%.4f ", nn.weights2.Data[0][j])
	}
	fmt.Println()

	fmt.Println("fc2.bias:")
	fmt.Printf("  %.4f\n", nn.bias2.Data[0][0])
}

// Calculate binary cross-entropy loss
func binaryCrossEntropy(predictions, targets *matrix.Matrix) float64 {
	if predictions.Rows != targets.Rows || predictions.Cols != targets.Cols {
		panic(fmt.Sprintf("Matrix dimensions don't match for BCE loss: predictions(%d,%d), targets(%d,%d)",
			predictions.Rows, predictions.Cols, targets.Rows, targets.Cols))
	}

	sum := 0.0
	n := float64(predictions.Rows * predictions.Cols)

	for i := 0; i < predictions.Rows; i++ {
		for j := 0; j < predictions.Cols; j++ {
			y := targets.Data[i][j]
			p := predictions.Data[i][j]

			// Avoid log(0)
			if p < 1e-15 {
				p = 1e-15
			} else if p > 1-1e-15 {
				p = 1 - 1e-15
			}

			sum += y*math.Log(p) + (1-y)*math.Log(1-p)
		}
	}

	return -sum / n
}

// --
// 生成特征索引的组合（包括不同次数）
func generateCombinations(featureCount, maxDegree int) [][]int {
	var combinations [][]int
	indices := make([]int, featureCount)
	for i := range indices {
		indices[i] = i
	}

	for d := 1; d <= maxDegree; d++ {
		combGen(indices, d, []int{}, &combinations)
	}
	return combinations
}

// 递归生成组合
func combGen(indices []int, degree int, current []int, result *[][]int) {
	if degree == 0 {
		combo := make([]int, len(current))
		copy(combo, current)
		*result = append(*result, combo)
		return
	}

	start := 0
	if len(current) > 0 {
		start = current[len(current)-1]
	}

	for i := start; i < len(indices); i++ {
		newCurrent := append(current, i)
		combGen(indices, degree-1, newCurrent, result)
	}
}
