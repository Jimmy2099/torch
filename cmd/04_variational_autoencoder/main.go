// main.go (Relevant VAE initialization part)
package main

import (
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"github.com/Jimmy2099/torch/layer"
	"github.com/Jimmy2099/torch/testing"
	"log"
	"math/rand"
	"os"
	"path/filepath"
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

type VAE struct {
	// Encoder layers
	encoderConv0  *torch.ConvLayer
	encoderConv1  *torch.BatchNormLayer
	encoderReLU2  *torch.ReLULayer // Assuming you have ReLU defined
	encoderConv3  *torch.ConvLayer
	encoderConv4  *torch.BatchNormLayer
	encoderReLU5  *torch.ReLULayer
	encoderConv6  *torch.ConvLayer
	encoderConv7  *torch.BatchNormLayer
	encoderReLU8  *torch.ReLULayer
	encoderConv9  *torch.ConvLayer
	encoderConv10 *torch.BatchNormLayer
	encoderReLU11 *torch.ReLULayer

	flatten  *torch.FlattenLayer
	fcMu     *torch.LinearLayer
	fcLogvar *torch.LinearLayer

	decoderFc *torch.LinearLayer

	decoderConv0  *layer.ConvTranspose2dLayer
	decoderConv1  *torch.BatchNormLayer
	decoderReLU2  *torch.ReLULayer
	decoderConv3  *layer.ConvTranspose2dLayer
	decoderConv4  *torch.BatchNormLayer
	decoderReLU5  *torch.ReLULayer
	decoderConv6  *layer.ConvTranspose2dLayer
	decoderConv7  *torch.BatchNormLayer
	decoderReLU8  *torch.ReLULayer
	decoderConv9  *layer.ConvTranspose2dLayer
	decoderTanh10 *layer.TanhLayer
}

type layerLoadInfo struct {
	pyTorchName string
	goLayer     torch.LayerLoader
	weightShape []int
	biasShape   []int
}

type Matrix struct {
	Data [][]float64
}

func NewVAE() *VAE {
	fmt.Println("Initializing VAE layers...")

	bnEps := 1e-5
	bnMomentum := 0.1

	vae := &VAE{
		//	  (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
		encoderConv0:  torch.NewConvLayer(3, 64, 5, 2, 2),
		encoderConv1:  torch.NewBatchNormLayer(64, bnEps, bnMomentum),
		encoderReLU2:  torch.NewReLULayer(),
		encoderConv3:  torch.NewConvLayer(64, 128, 5, 2, 2),
		encoderConv4:  torch.NewBatchNormLayer(128, bnEps, bnMomentum),
		encoderReLU5:  torch.NewReLULayer(),
		encoderConv6:  torch.NewConvLayer(128, 256, 5, 2, 2),
		encoderConv7:  torch.NewBatchNormLayer(256, bnEps, bnMomentum),
		encoderReLU8:  torch.NewReLULayer(),
		encoderConv9:  torch.NewConvLayer(256, 512, 5, 2, 2),
		encoderConv10: torch.NewBatchNormLayer(512, bnEps, bnMomentum),
		encoderReLU11: torch.NewReLULayer(),

		flatten:  torch.NewFlattenLayer(),
		fcMu:     torch.NewLinearLayer(8192, 64),
		fcLogvar: torch.NewLinearLayer(8192, 64),

		decoderFc: torch.NewLinearLayer(64, 8192),

		////	  (0): ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
		////	  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		////	  (2): ReLU(inplace=True)
		////	  (3): ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
		////	  (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		////	  (5): ReLU(inplace=True)
		////	  (6): ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
		////	  (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		////	  (8): ReLU(inplace=True)
		////	  (9): ConvTranspose2d(64, 3, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
		////	  (10): Tanh()
		decoderConv0: layer.NewConvTranspose2dLayer(
			512,  // 输入通道数 - 必须与前一层输出匹配
			256,  // 输出通道数
			5, 5, // 卷积核大小
			2, 2, // 步长
			2, 2, // 填充
			1, 1, // 输出填充
		),
		decoderConv1: torch.NewBatchNormLayer(256, bnEps, bnMomentum),
		decoderReLU2: torch.NewReLULayer(),
		decoderConv3: layer.NewConvTranspose2dLayer(
			256, 128,
			5, 5,
			2, 2,
			2, 2,
			1, 1,
		),
		decoderConv4: torch.NewBatchNormLayer(128, bnEps, bnMomentum),
		decoderReLU5: torch.NewReLULayer(),
		decoderConv6: layer.NewConvTranspose2dLayer(
			128, 64,
			5, 5,
			2, 2,
			2, 2,
			1, 1,
		),
		decoderConv7: torch.NewBatchNormLayer(64, bnEps, bnMomentum),
		decoderReLU8: torch.NewReLULayer(),
		decoderConv9: layer.NewConvTranspose2dLayer(
			64, 3,
			5, 5,
			2, 2,
			2, 2,
			1, 1,
		),
		decoderTanh10: layer.NewTanhLayer(),
	}
	fmt.Println("Layers initialized.")

	// --- Define the mapping --- Use correct PyTorch names ---
	loadInfos := []layerLoadInfo{
		// Encoder
		{"encoder_conv.0.weight", vae.encoderConv0, []int{64, 3, 5, 5}, nil},
		{"encoder_conv.0.bias", vae.encoderConv0, nil, []int{64}},
		{"encoder_conv.1.weight", vae.encoderConv1, []int{64}, nil},
		{"encoder_conv.1.bias", vae.encoderConv1, nil, []int{64}},
		{"encoder_conv.1.weight", vae.encoderConv1, []int{64}, nil},
		{"encoder_conv.1.bias", vae.encoderConv1, []int{64}, nil},
		{"encoder_conv.3.weight", vae.encoderConv3, []int{128, 64, 5, 5}, nil},
		{"encoder_conv.3.bias", vae.encoderConv3, nil, []int{128}},
		{"encoder_conv.4.weight", vae.encoderConv4, []int{128}, nil},
		{"encoder_conv.4.bias", vae.encoderConv4, nil, []int{128}},
		{"encoder_conv.4.weight", vae.encoderConv4, []int{128}, nil}, // Optional
		{"encoder_conv.4.bias", vae.encoderConv4, []int{128}, nil},   // Optional
		{"encoder_conv.6.weight", vae.encoderConv6, []int{256, 128, 5, 5}, nil},
		{"encoder_conv.6.bias", vae.encoderConv6, nil, []int{256}},
		{"encoder_conv.7.weight", vae.encoderConv7, []int{256}, nil},
		{"encoder_conv.7.bias", vae.encoderConv7, nil, []int{256}},
		{"encoder_conv.7.weight", vae.encoderConv7, []int{256}, nil}, // Optional
		{"encoder_conv.7.bias", vae.encoderConv7, []int{256}, nil},   // Optional
		{"encoder_conv.9.weight", vae.encoderConv9, []int{512, 256, 5, 5}, nil},
		{"encoder_conv.9.bias", vae.encoderConv9, nil, []int{512}},
		{"encoder_conv.10.weight", vae.encoderConv10, []int{512}, nil},
		{"encoder_conv.10.bias", vae.encoderConv10, nil, []int{512}},
		{"encoder_conv.10.weight", vae.encoderConv10, []int{512}, nil}, // Optional
		{"encoder_conv.10.bias", vae.encoderConv10, []int{512}, nil},   // Optional

		// FC Layers
		{"fc_mu.weight", vae.fcMu, []int{64, 8192}, nil},
		{"fc_mu.bias", vae.fcMu, nil, []int{64}},
		{"fc_logvar.weight", vae.fcLogvar, []int{64, 8192}, nil},
		{"fc_logvar.bias", vae.fcLogvar, nil, []int{64}},

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

		// Decoder Fully Connected
		// Layer: decoder_fc.weight, Shape: torch.Size([8192, 64]), dim: 2
		// Layer: decoder_fc.bias, Shape: torch.Size([8192]), dim: 1
		{"decoder_fc.weight", vae.decoderFc, []int{8192, 64}, nil},
		{"decoder_fc.bias", vae.decoderFc, nil, []int{8192}},

		// Layer: decoder_conv.0.weight, Shape: torch.Size([512, 256, 5, 5]), dim: 4
		// Layer: decoder_conv.0.bias, Shape: torch.Size([256]), dim: 1
		{"decoder_conv.0.weight", vae.decoderConv0, []int{512, 256, 5, 5}, nil},
		{"decoder_conv.0.bias", vae.decoderConv0, nil, []int{256}},
		// Layer: decoder_conv.1.weight, Shape: torch.Size([256]), dim: 1
		// Layer: decoder_conv.1.bias, Shape: torch.Size([256]), dim: 1
		{"decoder_conv.1.weight", vae.decoderConv1, []int{256}, nil},
		{"decoder_conv.1.bias", vae.decoderConv1, nil, []int{256}},
		// Layer: decoder_conv.3.weight, Shape: torch.Size([256, 128, 5, 5]), dim: 4
		// Layer: decoder_conv.3.bias, Shape: torch.Size([128]), dim: 1
		{"decoder_conv.3.weight", vae.decoderConv3, []int{256, 128, 5, 5}, nil},
		{"decoder_conv.3.bias", vae.decoderConv3, nil, []int{128}},
		// Layer: decoder_conv.4.weight, Shape: torch.Size([128]), dim: 1
		// Layer: decoder_conv.4.bias, Shape: torch.Size([128]), dim: 1
		{"decoder_conv.4.weight", vae.decoderConv4, []int{128}, nil},
		{"decoder_conv.4.bias", vae.decoderConv4, nil, []int{128}},
		// Layer: decoder_conv.6.weight, Shape: torch.Size([128, 64, 5, 5]), dim: 4
		// Layer: decoder_conv.6.bias, Shape: torch.Size([64]), dim: 1
		{"decoder_conv.6.weight", vae.decoderConv6, []int{128, 64, 5, 5}, nil},
		{"decoder_conv.6.bias", vae.decoderConv6, nil, []int{64}},
		// Layer: decoder_conv.7.weight, Shape: torch.Size([64]), dim: 1
		// Layer: decoder_conv.7.bias, Shape: torch.Size([64]), dim: 1
		{"decoder_conv.7.weight", vae.decoderConv7, []int{64}, nil},
		{"decoder_conv.7.bias", vae.decoderConv7, nil, []int{64}},
		// Layer: decoder_conv.9.weight, Shape: torch.Size([64, 3, 5, 5]), dim: 4
		// Layer: decoder_conv.9.bias, Shape: torch.Size([3]), dim: 1
		{"decoder_conv.9.weight", vae.decoderConv9, []int{64, 3, 5, 5}, nil},
		{"decoder_conv.9.bias", vae.decoderConv9, nil, []int{3}},
	}
	var weightsDir string
	{
		d, err := os.Getwd()
		if err != nil {
			panic(fmt.Sprint("Error getting working directory: %v\n", err))
		}
		weightsDir = filepath.Join(d, "py", "data")
		fmt.Printf("Looking for weights in: %s\n", weightsDir)

	}
	for i := 0; i < len(loadInfos); i++ {
		fmt.Println("loading layer", loadInfos[i].pyTorchName)
		if loadInfos[i].weightShape != nil {
			weightsFilePath := filepath.Join(weightsDir, loadInfos[i].pyTorchName+".csv")
			data, err := torch.LoadFlatDataFromCSV(weightsFilePath)
			if err != nil {
				panic(err)
			}
			loadInfos[i].goLayer.SetWeightsAndShape(data, loadInfos[i].weightShape)
			log.Println(weightsFilePath, loadInfos[i].weightShape)
		} else if loadInfos[i].biasShape != nil {
			biasFilePath := filepath.Join(weightsDir, loadInfos[i].pyTorchName+".csv")
			data, err := torch.LoadFlatDataFromCSV(biasFilePath)
			if err != nil {
				panic(err)
			}
			log.Println(biasFilePath, loadInfos[i].biasShape)
			loadInfos[i].goLayer.SetBiasAndShape(data, loadInfos[i].biasShape)
		}
	}

	fmt.Println("\n--- VAE model parameters loaded successfully. ---")
	return vae
}
func GenerateRandomNoise(numSamples, latentDim int) *tensor.Tensor {
	data := make([]float64, numSamples*latentDim)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return tensor.NewTensor(data, []int{numSamples, latentDim})
}

func main() {
	vae := NewVAE()
	log.Println("VAE model created and loaded.")
	log.Println(vae)
	x := GenerateRandomNoise(64, 64)
	x = vae.Decode(x)
	//x = x.Mul(tensor.NewTensor([]float64{0.5}, []int{1})).Add(tensor.NewTensor([]float64{0.5}, []int{1}))
	//x = x.Clamp(0, 1)
	x.Reshape([]int{1, len(x.Data)})
	x.SaveToCSV("./py/test.csv")
}

func (v *VAE) Encode(x *tensor.Tensor) *tensor.Tensor {
	fmt.Println("\n=== Starting VAE Forward Pass ===")
	fmt.Printf("Input shape: %v\n", x.Shape)

	// --- Encoder ---
	fmt.Println("\nEncoder Conv 0:")
	x = v.encoderConv0.Forward(x)
	fmt.Printf("After enc_conv0: %v\n", x.Shape)

	fmt.Println("Encoder BN 1:")
	x = v.encoderConv1.Forward(x)
	fmt.Printf("After enc_bn1: %v\n", x.Shape)

	fmt.Println("Encoder ReLU 2:")
	x = v.encoderReLU2.Forward(x)
	fmt.Printf("After enc_relu2: %v\n", x.Shape)

	fmt.Println("\nEncoder Conv 3:")
	x = v.encoderConv3.Forward(x)
	fmt.Printf("After enc_conv3: %v\n", x.Shape)

	fmt.Println("Encoder BN 4:")
	x = v.encoderConv4.Forward(x)
	fmt.Printf("After enc_bn4: %v\n", x.Shape)

	fmt.Println("Encoder ReLU 5:")
	x = v.encoderReLU5.Forward(x)
	fmt.Printf("After enc_relu5: %v\n", x.Shape)

	fmt.Println("\nEncoder Conv 6:")
	x = v.encoderConv6.Forward(x)
	fmt.Printf("After enc_conv6: %v\n", x.Shape)

	fmt.Println("Encoder BN 7:")
	x = v.encoderConv7.Forward(x)
	fmt.Printf("After enc_bn7: %v\n", x.Shape)

	fmt.Println("Encoder ReLU 8:")
	x = v.encoderReLU8.Forward(x)
	fmt.Printf("After enc_relu8: %v\n", x.Shape)

	fmt.Println("\nEncoder Conv 9:")
	x = v.encoderConv9.Forward(x)
	fmt.Printf("After enc_conv9: %v\n", x.Shape)

	fmt.Println("Encoder BN 10:")
	x = v.encoderConv10.Forward(x)
	fmt.Printf("After enc_bn10: %v\n", x.Shape)

	fmt.Println("Encoder ReLU 11:")
	x = v.encoderReLU11.Forward(x)
	fmt.Printf("After enc_relu11: %v\n", x.Shape)

	// --- Flatten ---
	fmt.Println("\nFlatten:")
	flatFeatures := v.flatten.Forward(x)
	fmt.Printf("After flatten: %v\n", flatFeatures.Shape)

	// --- Latent Variables ---
	fmt.Println("\nFC Mu:")
	mu := v.fcMu.Forward(flatFeatures)
	fmt.Printf("After fc_mu: %v\n", mu.Shape)

	fmt.Println("\nFC LogVar:")
	logvar := v.fcLogvar.Forward(flatFeatures)
	fmt.Printf("After fc_logvar: %v\n", logvar.Shape)
	return x
}

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

// nn.BatchNorm2d(gf_dim * 4)
// //	(decoder_conv): Sequential(
// //	  (0): ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
// //	  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
// //	  (2): ReLU(inplace=True)
// //	  (3): ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
// //	  (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
// //	  (5): ReLU(inplace=True)
// //	  (6): ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
// //	  (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
// //	  (8): ReLU(inplace=True)
// //	  (9): ConvTranspose2d(64, 3, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
// //	  (10): Tanh()
// //	)

func (v *VAE) Decode(x *tensor.Tensor) *tensor.Tensor {

	{
		// debug
		d, err := torch.LoadFlatDataFromCSV("./py/noise_1.csv")
		if err != nil {
			panic(err)
		}
		x = tensor.NewTensor(d, []int{64, 64})
	}
	{
		fmt.Println("\nDecoder FC:")
		x = v.decoderFc.Forward(x)
		fmt.Printf("After decoder_fc: %v\n", x.Shape)
		decoderReshapeChannels := 512
		decoderReshapeSize := 4
		// --- Reshape ---
		fmt.Println("\nReshape for Decoder Conv:")
		batchSize := x.Shape[0]
		x = x.Reshape([]int{
			batchSize,
			decoderReshapeChannels,
			decoderReshapeSize,
			decoderReshapeSize,
		})
		fmt.Printf("After reshape: %v\n", x.Shape)
	}

	//	{
	//		x = testing.GetPytorchInitData(fmt.Sprint(`
	//import time
	//import os
	//torch.manual_seed(int(time.time()))
	//out = torch.randn(64, 64)
	//fc_layer=torch.nn.Linear(64 ,8192)
	//fc_weight = np.loadtxt("./py/data/decoder_fc.weight.csv", delimiter=",")
	//
	//fc_layer.weight.data = torch.tensor(
	//	fc_weight,
	//	dtype=torch.float32
	//).to("cpu")
	//
	//fc_layer.bias.data = torch.tensor(
	//	np.loadtxt("./py/data/decoder_fc.bias.csv", delimiter=","),
	//	dtype=torch.float32
	//).to("cpu")
	//out = fc_layer(out)
	//print(out.shape)
	//out = out.view(-1 ,512 ,4 ,4)
	//`))
	//	}

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
			x = x.Mul(tensor.NewTensor([]float64{0.5}, []int{1})).Add(tensor.NewTensor([]float64{0.5}, []int{1}))
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

func (v *VAE) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = v.Encode(x)
	x = v.Decode(x)
	return x
}
