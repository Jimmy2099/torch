package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/optimizer"
)

func main() {
	rand.Seed(42)

	numSamples := 100
	xData := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		xData[i] = float32(rand.NormFloat64())
	}
	X := tensor.NewTensor(xData, []int{numSamples, 1})

	yData := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		yData[i] = float32(math.Sin(float64(xData[i])))
	}
	y := tensor.NewTensor(yData, []int{numSamples, 1})

	layer := torch.NewLinearLayer(1, 1)
	layer.SetWeights([]float32{0.5})
	layer.SetBias([]float32{0.0})

	fmt.Println("Initial Parameters:")
	fmt.Printf("Weight: %v\n", layer.Weights.Data)
	fmt.Printf("Bias: %v\n", layer.Bias.Data)

	optim := optimizer.NewSGD([]*tensor.Tensor{
		layer.Weights,
		layer.Bias,
	}, 0.01)

	numEpochs := 100
	for epoch := 0; epoch < numEpochs; epoch++ {
		outputs := layer.Forward(X)

		loss := outputs.LossMSE(y)

		layer.ZeroGrad()
		loss.Backward()
		optim.Step()

		if (epoch+1)%10 == 0 {
			fmt.Printf("Epoch [%d/%d], Loss: %.4f\n",
				epoch+1, numEpochs, loss.Data[0])
		}
	}

	fmt.Println("\nTrained Parameters:")
	fmt.Printf("Weight: %v\n", layer.Weights.Data)
	fmt.Printf("Bias: %v\n", layer.Bias.Data)

	numTest := 5
	xTestData := make([]float32, numTest)
	for i := 0; i < numTest; i++ {
		xTestData[i] = float32(rand.NormFloat64())
	}
	X_test := tensor.NewTensor(xTestData, []int{numTest, 1})

	yReal := make([]float32, numTest)
	for i := 0; i < numTest; i++ {
		yReal[i] = float32(math.Sin(float64(xTestData[i])))
	}
	predictions := layer.Forward(X_test)

	fmt.Println("\nTest Results:")
	fmt.Println("Real values:", yReal)
	fmt.Println("Predictions:", predictions.Data)
}
