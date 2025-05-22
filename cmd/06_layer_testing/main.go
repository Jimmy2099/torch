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

	fmt.Printf("W = %v\n", layer.Weights)
	fmt.Printf("B = %v\n", layer.Bias)

	optim := optimizer.NewSGD([]*tensor.Tensor{layer.Weights, layer.Bias}, 0.1)

	for i := 0; i < 100; i++ {
		out := layer.Forward(x)
		loss := out.LossMSE(y)
		loss.Backward()
		optim.Step()
		layer.Weights.ZeroGrad()
		layer.Bias.ZeroGrad()
		fmt.Printf("Epoch %d: loss = %v\n", i, loss.Data)
	}
	fmt.Printf("W after update: %v\n", layer.Weights)
	fmt.Printf("B after update: %v\n", layer.Bias)

}
