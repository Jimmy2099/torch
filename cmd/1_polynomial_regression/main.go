package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/optimizer"
)

type RegressionModel struct {
	Linear1 *torch.LinearLayer
	Relu    *torch.ReLULayer
	Linear2 *torch.LinearLayer
}

func NewRegressionModel(inputDim, hiddenDim int) *RegressionModel {
	return &RegressionModel{
		Linear1: torch.NewLinearLayer(inputDim, hiddenDim),
		Relu:    torch.NewReLULayer(),
		Linear2: torch.NewLinearLayer(hiddenDim, 1),
	}
}

func (m *RegressionModel) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = m.Linear1.Forward(x)
	x = m.Relu.Forward(x)
	return m.Linear2.Forward(x)
}

func (m *RegressionModel) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{
		m.Linear1.Weights,
		m.Linear1.Bias,
		m.Linear2.Weights,
		m.Linear2.Bias,
	}
}

var targetFunc = func(x1, x2 float32) float32 {
	return 3 + 2*x1 + 1.5*x2 +
		0.5*x1*x1 - 0.8*x2*x2 +
		0.3*x1*x1*x1
}

func generateData(numSamples int) (*tensor.Tensor, *tensor.Tensor) {
	rand.Seed(time.Now().UnixNano())

	X := make([]float32, numSamples*2)
	y := make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		x1 := rand.Float32()
		x2 := rand.Float32()
		X[i*2] = x1
		X[i*2+1] = x2
		y[i] = targetFunc(x1, x2)
	}

	return tensor.NewTensor(X, []int{numSamples, 2}),
		tensor.NewTensor(y, []int{numSamples, 1})
}

func trainModel(model *RegressionModel, X, y *tensor.Tensor, epochs int, lr float32) {
	optim := optimizer.NewSGD(model.Parameters(), lr)

	for epoch := 1; epoch <= epochs; epoch++ {
		pred := model.Forward(X)

		loss := pred.LossMSE(y)

		for _, p := range model.Parameters() {
			p.ZeroGrad()
		}

		loss.Backward()

		optim.Step()

		if epoch%10 == 0 || epoch == 1 {
			fmt.Printf("Epoch [%4d/%d] | Loss: %.4f\n",
				epoch, epochs, loss.Data[0])
		}
	}
}

func main() {
	X, y := generateData(1000)

	model := NewRegressionModel(2, 10)

	const (
		learningRate = 0.01
		totalEpochs  = 100
	)

	fmt.Println("=== Training Started ===")
	trainModel(model, X, y, totalEpochs, learningRate)
	fmt.Println("=== Training Completed ===")

	testX := tensor.NewTensor([]float32{0.2, 0.3}, []int{1, 2})

	pred := model.Forward(testX)
	actual := targetFunc(testX.Data[0], testX.Data[1])

	fmt.Printf("\nPrediction: %.4f\n", pred.Data[0])
	fmt.Printf("Actual Value: %.4f\n", actual)
	fmt.Printf("Absolute Error: %.4f\n", math.Abs(float64(pred.Data[0]-float32(actual))))
}
