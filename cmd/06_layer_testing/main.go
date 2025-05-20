package main

import (
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/optimizer"
	"math"
)

func main() {
	layer := torch.NewLinearLayer(2, 2)
	layer.SetWeights([]float32{0.5, 0.5, 0.5, 0.5})
	layer.SetBias([]float32{0, 0})

	x := tensor.NewTensor([]float32{1, 2}, []int{1, 2})
	y := tensor.NewTensor([]float32{0, 0}, []int{1, 2})
	for i := range x.Data {
		y.Data[i] = float32(math.Sin(float64(x.Data[i])))
	}

	out := layer.Forward(x)
	lossTensor := out.LossMSE(y)
	lossValue := lossTensor
	fmt.Printf("Loss before backward: %.6f\n", lossValue)

	layer.ZeroGrad()

	//TO DO
	//loss.Backward()

	W := layer.Weights
	B := layer.Bias

	fmt.Printf("dW = %v\n", W)
	fmt.Printf("db = %v\n", B)

	optim := optimizer.NewSGD([]*tensor.Tensor{W, B}, 0.1)
	optim.Step()

	fmt.Printf("W after update: %v\n", W.Data)
}
