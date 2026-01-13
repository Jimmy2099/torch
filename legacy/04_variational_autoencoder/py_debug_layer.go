package main

import (
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/testing"
)

func PyConvT(x *tensor.Tensor, v torch.LayerForTesting, inChannels, outChannels int) *tensor.Tensor {
	kernelSize := []int{5, 5}
	stride := []int{2, 2}
	padding := []int{2, 2}
	outputPadding := []int{1, 1}

	script := fmt.Sprintf(
		`torch.nn.ConvTranspose2d(%d, %d, kernel_size=(%d,%d), stride=(%d,%d), padding=(%d,%d), output_padding=(%d,%d))`,
		inChannels, outChannels, kernelSize[0], kernelSize[1],
		stride[0], stride[1], padding[0], padding[1],
		outputPadding[0], outputPadding[1],
	)
	x = testing.GetLayerTestResult32(script, v, x)
	return x
}

func PyBatchNorm2d(x *tensor.Tensor, v torch.LayerForTesting, num int) *tensor.Tensor {

	script := fmt.Sprintf(
		`torch.nn.BatchNorm2d(%d)`,
		num,
	)
	x = testing.GetLayerTestResult32(script, v, x)
	return x
}

func PyReLU(x *tensor.Tensor, v torch.LayerForTesting) *tensor.Tensor {

	script := fmt.Sprintf(
		`torch.nn.ReLU(True)`,
	)
	x = testing.GetLayerTestResult32(script, v, x)
	return x
}

func PyTanh(x *tensor.Tensor, v torch.LayerForTesting) *tensor.Tensor {

	script := fmt.Sprintf(
		`torch.nn.Tanh()`,
	)
	x = testing.GetLayerTestResult32(script, v, x)
	return x
}

func (v *VAE) PyDecode(x *tensor.Tensor) *tensor.Tensor {

	{
		x = testing.GetPytorchInitData(fmt.Sprint(`
	import time
	import os
	torch.manual_seed(int(time.time()))
	out = torch.randn(64, 64)
	fc_layer=torch.nn.Linear(64 ,8192)
	fc_weight = np.loadtxt("./py/data/decoder_fc.weight.csv", delimiter=",")
	
	fc_layer.weight.data = torch.tensor(
		fc_weight,
		dtype=torch.float32
	).to("cpu")
	
	fc_layer.bias.data = torch.tensor(
		np.loadtxt("./py/data/decoder_fc.bias.csv", delimiter=","),
		dtype=torch.float32
	).to("cpu")
	out = fc_layer(out)
	print(out.shape)
	out = out.view(-1 ,512 ,4 ,4)
	`))
	}

	{
		fmt.Println("\nDecoder ConvT 0:")
		x = v.decoderConv0.Forward(x)
		fmt.Printf("After dec_convT0: %v\n", x.GetShape())

		fmt.Println("Decoder BN 1:")
		x = v.decoderConv1.Forward(x)
		fmt.Printf("After dec_bn1: %v\n", x.GetShape())

		fmt.Println("Decoder ReLU 2:")
		x = v.decoderReLU2.Forward(x)
		fmt.Printf("After dec_relu2: %v\n", x.GetShape())

		fmt.Println("\nDecoder ConvT 3:")
		x = v.decoderConv3.Forward(x)
		fmt.Printf("After dec_convT3: %v\n", x.GetShape())

		fmt.Println("Decoder BN 4:")
		x = v.decoderConv4.Forward(x)
		fmt.Printf("After dec_bn4: %v\n", x.GetShape())

		fmt.Println("Decoder ReLU 5:")
		x = v.decoderReLU5.Forward(x)
		fmt.Printf("After dec_relu5: %v\n", x.GetShape())

		fmt.Println("\nDecoder ConvT 6:")
		x = v.decoderConv6.Forward(x)
		fmt.Printf("After dec_convT6: %v\n", x.GetShape())

		fmt.Println("Decoder BN 7:")
		x = v.decoderConv7.Forward(x)
		fmt.Printf("After dec_bn7: %v\n", x.GetShape())

		fmt.Println("Decoder ReLU 8:")
		x = v.decoderReLU8.Forward(x)
		fmt.Printf("After dec_relu8: %v\n", x.GetShape())

		fmt.Println("\nDecoder ConvT 9:")
		x = v.decoderConv9.Forward(x)
		fmt.Printf("After dec_convT9: %v\n", x.GetShape())

		fmt.Println("Decoder Tanh 10 (Output):")
		x = v.decoderTanh10.Forward(x)
		fmt.Printf("After dec_tanh10 (output): %v\n", x.GetShape())

		{
			x = x.Mul(tensor.NewTensor([]float32{0.5}, []int{1})).Add(tensor.NewTensor([]float32{0.5}, []int{1}))
			x = x.Clamp(0, 1)
		}

		testing.GetTensorTestResult(`
sv_image(in1)
    out=in1
`, x, x)
		fmt.Println("\n=== VAE Forward Pass Complete ===")
	}
	return x
}
