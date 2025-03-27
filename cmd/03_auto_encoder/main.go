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

func NewAutoEncoder() *AutoEncoder {
	return &AutoEncoder{
		fc1:   torch.NewLinearLayer(784, 128),
		relu1: torch.NewReLULayer(),
		fc2:   torch.NewLinearLayer(128, 64),
		relu2: torch.NewReLULayer(),
		fc3:   torch.NewLinearLayer(64, 32),

		//--

		fc4:      torch.NewLinearLayer(32, 64),
		relu3:    torch.NewReLULayer(),
		fc5:      torch.NewLinearLayer(64, 128),
		relu4:    torch.NewReLULayer(),
		fc6:      torch.NewLinearLayer(128, 784),
		Sigmoid1: torch.NewSigmoidLayer(),
	}
}

func (ae *AutoEncoder) Forward(x *tensor.Tensor) *tensor.Tensor {
	fmt.Println("\n=== Starting AutoEncoder Forward Pass ===")
	fmt.Printf("Input shape: %v\n", x.Shape)

	// 编码器部分
	fmt.Println("\nEncoder FC1:")
	x = ae.fc1.Forward(x)
	fmt.Printf("After fc1: %v\n", x.Shape)

	fmt.Println("\nReLU1:")
	x = ae.relu1.Forward(x)
	fmt.Printf("After relu1: %v\n", x.Shape)

	fmt.Println("\nEncoder FC2:")
	x = ae.fc2.Forward(x)
	fmt.Printf("After fc2: %v\n", x.Shape)

	fmt.Println("\nReLU2:")
	x = ae.relu2.Forward(x)
	fmt.Printf("After relu2: %v\n", x.Shape)

	fmt.Println("\nEncoder FC3:")
	x = ae.fc3.Forward(x)
	fmt.Printf("After fc3: %v\n", x.Shape)

	// 解码器部分
	fmt.Println("\nDecoder FC4:")
	x = ae.fc4.Forward(x)
	fmt.Printf("After fc4: %v\n", x.Shape)

	fmt.Println("\nReLU3:")
	x = ae.relu3.Forward(x)
	fmt.Printf("After relu3: %v\n", x.Shape)

	fmt.Println("\nDecoder FC5:")
	x = ae.fc5.Forward(x)
	fmt.Printf("After fc5: %v\n", x.Shape)

	fmt.Println("\nReLU4:")
	x = ae.relu4.Forward(x)
	fmt.Printf("After relu4: %v\n", x.Shape)

	fmt.Println("\nDecoder FC6:")
	x = ae.fc6.Forward(x)
	fmt.Printf("After fc6: %v\n", x.Shape)

	fmt.Println("\nSigmoid:")
	x = ae.Sigmoid1.Forward(x)
	fmt.Printf("After sigmoid: %v\n", x.Shape)

	fmt.Println("\n=== AutoEncoder Forward Pass Complete ===")
	return x
}

func (ae *AutoEncoder) Parameters() []*tensor.Tensor {
	params := make([]*tensor.Tensor, 0)
	params = append(params, ae.fc1.Weights, ae.fc1.Bias)
	params = append(params, ae.fc2.Weights, ae.fc2.Bias)
	params = append(params, ae.fc3.Weights, ae.fc3.Bias)
	params = append(params, ae.fc4.Weights, ae.fc4.Bias)
	params = append(params, ae.fc5.Weights, ae.fc5.Bias)
	params = append(params, ae.fc6.Weights, ae.fc6.Bias)
	return params
}

func (ae *AutoEncoder) ZeroGrad() {
	ae.fc1.ZeroGrad()
	ae.fc2.ZeroGrad()
	ae.fc3.ZeroGrad()
	ae.fc4.ZeroGrad()
	ae.fc5.ZeroGrad()
	ae.fc6.ZeroGrad()
}

func main() {
	// 创建AutoEncoder模型
	model := NewAutoEncoder()

	// 示例输入 (784维向量，模拟MNIST图像)
	input := tensor.NewTensor(make([]float64, 784), []int{1, 784})

	// 前向传播
	output := model.Forward(input)
	fmt.Println("Output shape:", output.Shape)
}
