package main

import (
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"testing"
)

func TestLinearBackward(t *testing.T) {
	layer := torch.NewLinearLayer(2, 2)

	layer.SetWeights([]float32{0.5, 0.5, 0.5, 0.5})
	layer.SetBias([]float32{0.0, 0.0})

	x := tensor.NewTensor([]float32{1.0, 2.0}, []int{1, 2})
	y := tensor.NewTensor([]float32{1.0, 2.0}, []int{1, 2})
	for i := 0; i < len(x.Data); i++ {
		y.Data[i] = math.Sin(x.Data[i])
	}
	output := layer.Forward(x)
	loss := output.LossMSE(y)
	fmt.Println(loss)

}
