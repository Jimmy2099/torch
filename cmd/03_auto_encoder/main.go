package main

import (
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/tensor"
)

// AutoEncoder 	//Autoencoder(
//
//	//  (encoder): Sequential(
//	//    (0): Linear(in_features=784, out_features=128, bias=True)
//	//    (1): ReLU()
//	//    (2): Linear(in_features=128, out_features=64, bias=True)
//	//    (3): ReLU()
//	//    (4): Linear(in_features=64, out_features=32, bias=True)
//	//  )
//	//  (decoder): Sequential(
//	//    (0): Linear(in_features=32, out_features=64, bias=True)
//	//    (1): ReLU()
//	//    (2): Linear(in_features=64, out_features=128, bias=True)
//	//    (3): ReLU()
//	//    (4): Linear(in_features=128, out_features=784, bias=True)
//	//    (5): Sigmoid()
//	//  )
//	//)
type AutoEncoder struct {
	conv1 *torch.ConvLayer
	conv2 *torch.ConvLayer
	relu  *torch.ReLULayer
	pool  *torch.MaxPool2DLayer
	//编码器层
	fc1   *torch.LinearLayer
	relu1 *torch.ReLULayer
	fc2   *torch.LinearLayer
	relu2 *torch.ReLULayer
	fc3   *torch.LinearLayer

	//解码器层
	fc4      *torch.LinearLayer
	relu3    *torch.ReLULayer
	fc5      *torch.LinearLayer
	relu4    *torch.ReLULayer
	fc6      *torch.LinearLayer
	Sigmoid1 *torch.SigmoidLayer
}

// Forward performs the forward pass of the CNN.
func (c *AutoEncoder) Forward(x *tensor.Tensor) *tensor.Tensor {
	return nil
}

// TODO
func main() {
	model := &AutoEncoder{}
	output := model.Forward(nil)
	fmt.Println(output)

}
