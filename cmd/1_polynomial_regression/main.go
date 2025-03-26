package main

import (
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"math"
	"math/rand"
	"time"
)

// Neural Network implementation
type NeuralNetwork struct {
	Layers []torch.Layer
}

func (nn *NeuralNetwork) Parameters() []*tensor.Tensor {
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
			nn.Layers = append(nn.Layers, torch.NewReLULayer())
		}
	}

	return nn
}

// Forward performs forward pass through the neural network
func (nn *NeuralNetwork) Forward(input *tensor.Tensor) *tensor.Tensor {
	output := input
	for _, layer := range nn.Layers {
		output = layer.Forward(output)
	}
	return output
}

// PolynomialFeatures generates polynomial features from the input
func polynomialFeatures(X *tensor.Tensor, degree int) *tensor.Tensor {
	// 实现多项式特征生成
	features := make([]float64, 0)
	for d := 1; d <= degree; d++ {
		for i := 0; i < X.Shape[0]; i++ {
			val := X.Data[i]
			features = append(features, math.Pow(val, float64(d)))
		}
	}
	return tensor.NewTensor(features, []int{len(features)})
}

// Backward performs backward pass through the neural network
func (nn *NeuralNetwork) Backward(targets *tensor.Tensor, learningRate float64) {
	// Compute MSE loss gradient
	lastLayer := nn.Layers[len(nn.Layers)-1]
	output := lastLayer.(*torch.LinearLayer).Output

	// 确保输出形状与目标形状完全一致
	if !tensor.ShapeEqual(output.Shape, targets.Shape) {
		output = output.Reshape(targets.Shape)
	}

	// dL/dY = (Y - T) * 2/n
	gradOutput := tensor.Subtract(output, targets)
	batchSize := float64(targets.Shape[0])
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

func targetFunc(x1, x2 float64) float64 {
	return 3 + 2*x1 + 1.5*x2 + 0.5*math.Pow(x1, 2) - 0.8*math.Pow(x2, 2) + 0.3*math.Pow(x1, 3)
}

// ... existing code ...

func main() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Generate training data (10 samples, 2 features each)
	X_data := make([]float64, 2*10)
	for i := 0; i < 2*10; i++ {
		X_data[i] = rand.Float64() //[0.0,1.0)
	}
	X_train := tensor.NewTensor(X_data, []int{2, 10})

	// Define the target function
	y_data := make([]float64, 10)
	for j := 0; j < 10; j++ {
		x1 := X_train.Data[j*2]
		x2 := X_train.Data[j*2+1]
		y_data[j] = targetFunc(x1, x2)
	}
	y_train := tensor.NewTensor(y_data, []int{10})

	// Generate polynomial features (degree=3)
	degree := 3
	X_train_poly := polynomialFeatures(X_train, degree)
	X_train_poly = X_train_poly.Reshape([]int{degree * X_train.Shape[0], X_train.Shape[1]})

	// Print dimensions - 使用Shape代替Rows/Cols
	fmt.Printf("X_train dimensions: %v\n", X_train.Shape)
	fmt.Printf("X_train_poly dimensions: %v\n", X_train_poly.Shape)
	fmt.Printf("y_train dimensions: %v\n", y_train.Shape)

	// Create neural network [input -> 10 -> 1]
	inputDim := X_train_poly.Shape[0]
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
	test_data := []float64{0.2, 0.3} // 测试数据
	test_sample := tensor.NewTensor(test_data, []int{2, 1})
	test_sample_poly := polynomialFeatures(test_sample, degree)
	prediction := model.Forward(test_sample_poly)

	// Test model prediction performance on data range [0.0,1.0)
	fmt.Printf("\nPredicted value: %.4f\n", prediction.Data[0])

	// Calculate true value
	trueValue := targetFunc(test_data[0], test_data[1])
	fmt.Printf("True value: %.4f\n", trueValue)

	// Print error
	fmt.Printf("Error: %.4f\n", math.Abs(prediction.Data[0]-trueValue))
}
