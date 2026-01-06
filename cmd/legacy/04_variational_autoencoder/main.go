package main

import (
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/layer"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/pkg/log"
	"github.com/Jimmy2099/torch/testing"
	"os"
	"path/filepath"
)

type VAE struct {
	encoderConv0  *torch.ConvLayer
	encoderConv1  *torch.BatchNormLayer
	encoderReLU2  *torch.ReLULayer
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
	fcLogVar *torch.LinearLayer

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
	Data [][]float32
}

func NewVAE() *VAE {
	fmt.Println("Initializing VAE layers...")

	bnEps := float32(1e-5)
	bnMomentum := float32(0.1)

	vae := &VAE{
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
		fcLogVar: torch.NewLinearLayer(8192, 64),

		decoderFc: torch.NewLinearLayer(64, 8192),

		decoderConv0: layer.NewConvTranspose2dLayer(
			512,
			256,
			5, 5,
			2, 2,
			2, 2,
			1, 1,
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

	loadInfos := []layerLoadInfo{
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
		{"encoder_conv.4.weight", vae.encoderConv4, []int{128}, nil},
		{"encoder_conv.4.bias", vae.encoderConv4, []int{128}, nil},
		{"encoder_conv.6.weight", vae.encoderConv6, []int{256, 128, 5, 5}, nil},
		{"encoder_conv.6.bias", vae.encoderConv6, nil, []int{256}},
		{"encoder_conv.7.weight", vae.encoderConv7, []int{256}, nil},
		{"encoder_conv.7.bias", vae.encoderConv7, nil, []int{256}},
		{"encoder_conv.7.weight", vae.encoderConv7, []int{256}, nil},
		{"encoder_conv.7.bias", vae.encoderConv7, []int{256}, nil},
		{"encoder_conv.9.weight", vae.encoderConv9, []int{512, 256, 5, 5}, nil},
		{"encoder_conv.9.bias", vae.encoderConv9, nil, []int{512}},
		{"encoder_conv.10.weight", vae.encoderConv10, []int{512}, nil},
		{"encoder_conv.10.bias", vae.encoderConv10, nil, []int{512}},
		{"encoder_conv.10.weight", vae.encoderConv10, []int{512}, nil},
		{"encoder_conv.10.bias", vae.encoderConv10, []int{512}, nil},

		{"fc_mu.weight", vae.fcMu, []int{64, 8192}, nil},
		{"fc_mu.bias", vae.fcMu, nil, []int{64}},
		{"fc_logvar.weight", vae.fcLogVar, []int{64, 8192}, nil},
		{"fc_logvar.bias", vae.fcLogVar, nil, []int{64}},

		{"decoder_fc.weight", vae.decoderFc, []int{8192, 64}, nil},
		{"decoder_fc.bias", vae.decoderFc, nil, []int{8192}},

		{"decoder_conv.0.weight", vae.decoderConv0, []int{512, 256, 5, 5}, nil},
		{"decoder_conv.0.bias", vae.decoderConv0, nil, []int{256}},
		{"decoder_conv.1.weight", vae.decoderConv1, []int{256}, nil},
		{"decoder_conv.1.bias", vae.decoderConv1, nil, []int{256}},
		{"decoder_conv.3.weight", vae.decoderConv3, []int{256, 128, 5, 5}, nil},
		{"decoder_conv.3.bias", vae.decoderConv3, nil, []int{128}},
		{"decoder_conv.4.weight", vae.decoderConv4, []int{128}, nil},
		{"decoder_conv.4.bias", vae.decoderConv4, nil, []int{128}},
		{"decoder_conv.6.weight", vae.decoderConv6, []int{128, 64, 5, 5}, nil},
		{"decoder_conv.6.bias", vae.decoderConv6, nil, []int{64}},
		{"decoder_conv.7.weight", vae.decoderConv7, []int{64}, nil},
		{"decoder_conv.7.bias", vae.decoderConv7, nil, []int{64}},
		{"decoder_conv.9.weight", vae.decoderConv9, []int{64, 3, 5, 5}, nil},
		{"decoder_conv.9.bias", vae.decoderConv9, nil, []int{3}},
	}
	var weightsDir string
	{
		d, err := os.Getwd()
		if err != nil {
			panic(fmt.Sprintln("Error getting working directory:", err))
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

func main() {
	vae := NewVAE()
	log.Println("VAE model created and loaded.")
	x := tensor.RandomNormal([]int{64, 64})
	x = vae.Decode(x)
	x.Reshape([]int{1, len(x.Data)})
	x.SaveToCSV("./py/test.csv")
}

func (v *VAE) Encode(x *tensor.Tensor) *tensor.Tensor {
	fmt.Println("\n=== Starting VAE Forward Pass ===")
	fmt.Printf("Input shape: %v\n", x.GetShape())

	fmt.Println("\nEncoder Conv 0:")
	x = v.encoderConv0.Forward(x)
	fmt.Printf("After enc_conv0: %v\n", x.GetShape())

	fmt.Println("Encoder BN 1:")
	x = v.encoderConv1.Forward(x)
	fmt.Printf("After enc_bn1: %v\n", x.GetShape())

	fmt.Println("Encoder ReLU 2:")
	x = v.encoderReLU2.Forward(x)
	fmt.Printf("After enc_relu2: %v\n", x.GetShape())

	fmt.Println("\nEncoder Conv 3:")
	x = v.encoderConv3.Forward(x)
	fmt.Printf("After enc_conv3: %v\n", x.GetShape())

	fmt.Println("Encoder BN 4:")
	x = v.encoderConv4.Forward(x)
	fmt.Printf("After enc_bn4: %v\n", x.GetShape())

	fmt.Println("Encoder ReLU 5:")
	x = v.encoderReLU5.Forward(x)
	fmt.Printf("After enc_relu5: %v\n", x.GetShape())

	fmt.Println("\nEncoder Conv 6:")
	x = v.encoderConv6.Forward(x)
	fmt.Printf("After enc_conv6: %v\n", x.GetShape())

	fmt.Println("Encoder BN 7:")
	x = v.encoderConv7.Forward(x)
	fmt.Printf("After enc_bn7: %v\n", x.GetShape())

	fmt.Println("Encoder ReLU 8:")
	x = v.encoderReLU8.Forward(x)
	fmt.Printf("After enc_relu8: %v\n", x.GetShape())

	fmt.Println("\nEncoder Conv 9:")
	x = v.encoderConv9.Forward(x)
	fmt.Printf("After enc_conv9: %v\n", x.GetShape())

	fmt.Println("Encoder BN 10:")
	x = v.encoderConv10.Forward(x)
	fmt.Printf("After enc_bn10: %v\n", x.GetShape())

	fmt.Println("Encoder ReLU 11:")
	x = v.encoderReLU11.Forward(x)
	fmt.Printf("After enc_relu11: %v\n", x.GetShape())

	fmt.Println("\nFlatten:")
	flatFeatures := v.flatten.Forward(x)
	fmt.Printf("After flatten: %v\n", flatFeatures.GetShape())

	fmt.Println("\nFC Mu:")
	mu := v.fcMu.Forward(flatFeatures)
	fmt.Printf("After fc_mu: %v\n", mu.GetShape())

	fmt.Println("\nFC LogVar:")
	logvar := v.fcLogVar.Forward(flatFeatures)
	fmt.Printf("After fc_logvar: %v\n", logvar.GetShape())
	return x
}

func (v *VAE) Decode(x *tensor.Tensor) *tensor.Tensor {

	{
		fmt.Println("\nDecoder FC:")
		x = v.decoderFc.Forward(x)
		fmt.Printf("After decoder_fc: %v\n", x.GetShape())
		decoderReshapeChannels := 512
		decoderReshapeSize := 4
		fmt.Println("\nReshape for Decoder Conv:")
		batchSize := x.GetShape()[0]
		x = x.Reshape([]int{
			batchSize,
			decoderReshapeChannels,
			decoderReshapeSize,
			decoderReshapeSize,
		})
		fmt.Printf("After reshape: %v\n", x.GetShape())
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

func (v *VAE) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = v.Encode(x)
	x = v.Decode(x)
	return x
}
