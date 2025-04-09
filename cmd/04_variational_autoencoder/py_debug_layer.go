package main

import (
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/testing"
)

//	(encoder_conv): Sequential(
//	  (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
//	  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (2): ReLU(inplace=True)
//	  (3): Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
//	  (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (5): ReLU(inplace=True)
//	  (6): Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
//	  (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (8): ReLU(inplace=True)
//	  (9): Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
//	  (10): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (11): ReLU(inplace=True)
//	)
//	(flatten): Flatten(start_dim=1, end_dim=-1)
//	(fc_mu): Linear(in_features=8192, out_features=64, bias=True)
//	(fc_logvar): Linear(in_features=8192, out_features=64, bias=True)
//	(decoder_fc): Linear(in_features=64, out_features=8192, bias=True)
//	(decoder_conv): Sequential(
//	  (0): ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
//	  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (2): ReLU(inplace=True)
//	  (3): ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
//	  (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (5): ReLU(inplace=True)
//	  (6): ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
//	  (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (8): ReLU(inplace=True)
//	  (9): ConvTranspose2d(64, 3, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
//	  (10): Tanh()
//	)
//
// )
// Layer: encoder_conv.0.weight, Shape: torch.Size([64, 3, 5, 5]), dim: 4
// Layer: encoder_conv.0.bias, Shape: torch.Size([64]), dim: 1
// Layer: encoder_conv.1.weight, Shape: torch.Size([64]), dim: 1
// Layer: encoder_conv.1.bias, Shape: torch.Size([64]), dim: 1
// Layer: encoder_conv.3.weight, Shape: torch.Size([128, 64, 5, 5]), dim: 4
// Layer: encoder_conv.3.bias, Shape: torch.Size([128]), dim: 1
// Layer: encoder_conv.4.weight, Shape: torch.Size([128]), dim: 1
// Layer: encoder_conv.4.bias, Shape: torch.Size([128]), dim: 1
// Layer: encoder_conv.6.weight, Shape: torch.Size([256, 128, 5, 5]), dim: 4
// Layer: encoder_conv.6.bias, Shape: torch.Size([256]), dim: 1
// Layer: encoder_conv.7.weight, Shape: torch.Size([256]), dim: 1
// Layer: encoder_conv.7.bias, Shape: torch.Size([256]), dim: 1
// Layer: encoder_conv.9.weight, Shape: torch.Size([512, 256, 5, 5]), dim: 4
// Layer: encoder_conv.9.bias, Shape: torch.Size([512]), dim: 1
// Layer: encoder_conv.10.weight, Shape: torch.Size([512]), dim: 1
// Layer: encoder_conv.10.bias, Shape: torch.Size([512]), dim: 1
// Layer: fc_mu.weight, Shape: torch.Size([64, 8192]), dim: 2
// Layer: fc_mu.bias, Shape: torch.Size([64]), dim: 1
// Layer: fc_logvar.weight, Shape: torch.Size([64, 8192]), dim: 2
// Layer: fc_logvar.bias, Shape: torch.Size([64]), dim: 1
// Layer: decoder_fc.weight, Shape: torch.Size([8192, 64]), dim: 2
// Layer: decoder_fc.bias, Shape: torch.Size([8192]), dim: 1
// Layer: decoder_conv.0.weight, Shape: torch.Size([512, 256, 5, 5]), dim: 4
// Layer: decoder_conv.0.bias, Shape: torch.Size([256]), dim: 1
// Layer: decoder_conv.1.weight, Shape: torch.Size([256]), dim: 1
// Layer: decoder_conv.1.bias, Shape: torch.Size([256]), dim: 1
// Layer: decoder_conv.3.weight, Shape: torch.Size([256, 128, 5, 5]), dim: 4
// Layer: decoder_conv.3.bias, Shape: torch.Size([128]), dim: 1
// Layer: decoder_conv.4.weight, Shape: torch.Size([128]), dim: 1
// Layer: decoder_conv.4.bias, Shape: torch.Size([128]), dim: 1
// Layer: decoder_conv.6.weight, Shape: torch.Size([128, 64, 5, 5]), dim: 4
// Layer: decoder_conv.6.bias, Shape: torch.Size([64]), dim: 1
// Layer: decoder_conv.7.weight, Shape: torch.Size([64]), dim: 1
// Layer: decoder_conv.7.bias, Shape: torch.Size([64]), dim: 1
// Layer: decoder_conv.9.weight, Shape: torch.Size([64, 3, 5, 5]), dim: 4
// Layer: decoder_conv.9.bias, Shape: torch.Size([3]), dim: 1

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

	//{
	//	// debug
	//	d, err := torch.LoadFlatDataFromCSV("./py/noise_1.csv")
	//	if err != nil {
	//		panic(err)
	//	}
	//	x = tensor.NewTensor(d, []int{64, 64})
	//}
	//{
	//	fmt.Println("\nDecoder FC:")
	//	x = v.decoderFc.Forward(x)
	//	fmt.Printf("After decoder_fc: %v\n", x.Shape)
	//	decoderReshapeChannels := 512
	//	decoderReshapeSize := 4
	//	// --- Reshape ---
	//	fmt.Println("\nReshape for Decoder Conv:")
	//	batchSize := x.Shape[0]
	//	x = x.Reshape([]int{
	//		batchSize,
	//		decoderReshapeChannels,
	//		decoderReshapeSize,
	//		decoderReshapeSize,
	//	})
	//	fmt.Printf("After reshape: %v\n", x.Shape)
	//}

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

	//Decode
	{
		fmt.Println("\nDecoder ConvT 0:")
		x = v.decoderConv0.Forward(x)
		//x = PyConvT(x, v.decoderConv0, 512, 256)
		fmt.Printf("After dec_convT0: %v\n", x.Shape)

		fmt.Println("Decoder BN 1:")
		//x = PyBatchNorm2d(x, v.decoderConv1, 64*4)
		x = v.decoderConv1.Forward(x)
		fmt.Printf("After dec_bn1: %v\n", x.Shape)

		fmt.Println("Decoder ReLU 2:")
		x = v.decoderReLU2.Forward(x)
		//x = PyReLU(x, v.decoderReLU2)
		fmt.Printf("After dec_relu2: %v\n", x.Shape)

		fmt.Println("\nDecoder ConvT 3:")
		x = v.decoderConv3.Forward(x)
		//x = PyConvT(x, v.decoderConv3, 128, 64)
		fmt.Printf("After dec_convT3: %v\n", x.Shape)

		fmt.Println("Decoder BN 4:")
		//x = PyBatchNorm2d(x, v.decoderConv4, 64*2)
		x = v.decoderConv4.Forward(x)
		fmt.Printf("After dec_bn4: %v\n", x.Shape)

		fmt.Println("Decoder ReLU 5:")
		x = v.decoderReLU5.Forward(x)
		//x = PyReLU(x, v.decoderReLU5)
		fmt.Printf("After dec_relu5: %v\n", x.Shape)

		fmt.Println("\nDecoder ConvT 6:")
		x = v.decoderConv6.Forward(x)
		//x = PyConvT(x, v.decoderConv6, 128, 64)
		fmt.Printf("After dec_convT6: %v\n", x.Shape)

		fmt.Println("Decoder BN 7:")
		//x = PyBatchNorm2d(x, v.decoderConv7, 64)
		x = v.decoderConv7.Forward(x)
		fmt.Printf("After dec_bn7: %v\n", x.Shape)

		fmt.Println("Decoder ReLU 8:")
		x = v.decoderReLU8.Forward(x)
		//x = PyReLU(x, v.decoderReLU8)
		fmt.Printf("After dec_relu8: %v\n", x.Shape)

		fmt.Println("\nDecoder ConvT 9:")
		x = v.decoderConv9.Forward(x)
		//x = PyConvT(x, v.decoderConv9, 64, 3)
		fmt.Printf("After dec_convT9: %v\n", x.Shape)

		fmt.Println("Decoder Tanh 10 (Output):")
		x = v.decoderTanh10.Forward(x)
		//x = PyTanh(x, v.decoderTanh10)
		fmt.Printf("After dec_tanh10 (output): %v\n", x.Shape)

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
	//[64,3,64,64]
	return x
}
