// main.go (Relevant VAE initialization part)
package main

import (
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/layer"
	"os"
	"path/filepath"
	"strings"
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

func LoadMatrixFromCSV(filePath string) (*Matrix, error) {
	// Implement your CSV loading logic here
	// This is just a placeholder
	fmt.Printf("Placeholder: LoadMatrixFromCSV called for %s\n", filePath)
	// In a real scenario, you'd open the file, parse CSV, convert to float64
	// and handle potential errors.
	// Returning dummy data for structure:
	dummyData := [][]float64{{0.0}}            // Adjust size based on expected file content
	if strings.Contains(filePath, "running") { // Handle potentially missing optional files
		// _, err := os.Stat(filePath)
		// if os.IsNotExist(err) {
		// 	 return nil, fmt.Errorf("optional file not found") // Or return nil, nil
		// }
		// For placeholder, assume it might be missing
		// return nil, fmt.Errorf("optional file placeholder")
	}
	return &Matrix{Data: dummyData}, nil
	// return nil, fmt.Errorf("not implemented")
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

		decoderConv0:  layer.NewConvTranspose2dLayer(512, 256, 5, 5, 2, 2, 2, 2, 1, 1),
		decoderConv1:  torch.NewBatchNormLayer(256, bnEps, bnMomentum),
		decoderReLU2:  torch.NewReLULayer(),
		decoderConv3:  layer.NewConvTranspose2dLayer(256, 128, 5, 5, 2, 2, 2, 2, 1, 1),
		decoderConv4:  torch.NewBatchNormLayer(128, bnEps, bnMomentum),
		decoderReLU5:  torch.NewReLULayer(),
		decoderConv6:  layer.NewConvTranspose2dLayer(128, 64, 5, 5, 2, 2, 2, 2, 1, 1),
		decoderConv7:  torch.NewBatchNormLayer(64, bnEps, bnMomentum),
		decoderReLU8:  torch.NewReLULayer(),
		decoderConv9:  layer.NewConvTranspose2dLayer(64, 3, 5, 5, 2, 2, 2, 2, 1, 1),
		decoderTanh10: layer.NewTanhLayer(),
	}
	fmt.Println("Layers initialized.")

	// --- Define the mapping --- Use correct PyTorch names ---
	// (Make sure these match your Python model structure exactly)
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

		// Decoder
		{"decoder_fc.weight", vae.decoderFc, []int{8192, 64}, nil},
		{"decoder_fc.bias", vae.decoderFc, nil, []int{8192}},
		{"decoder_conv.0.weight", vae.decoderConv0, []int{512, 256, 5, 5}, nil}, // Note: ConvT weights are [in, out, k, k] in PyTorch - Check Go impl!
		{"decoder_conv.0.bias", vae.decoderConv0, nil, []int{256}},
		{"decoder_conv.1.weight", vae.decoderConv1, []int{256}, nil},
		{"decoder_conv.1.bias", vae.decoderConv1, nil, []int{256}},
		{"decoder_conv.1.weight", vae.decoderConv1, []int{256}, nil}, // Optional
		{"decoder_conv.1.bias", vae.decoderConv1, []int{256}, nil},   // Optional
		{"decoder_conv.3.weight", vae.decoderConv3, []int{256, 128, 5, 5}, nil},
		{"decoder_conv.3.bias", vae.decoderConv3, nil, []int{128}},
		{"decoder_conv.4.weight", vae.decoderConv4, []int{128}, nil},
		{"decoder_conv.4.bias", vae.decoderConv4, nil, []int{128}},
		{"decoder_conv.4.weight", vae.decoderConv4, []int{128}, nil}, // Optional
		{"decoder_conv.4.bias", vae.decoderConv4, []int{128}, nil},   // Optional
		{"decoder_conv.6.weight", vae.decoderConv6, []int{128, 64, 5, 5}, nil},
		{"decoder_conv.6.bias", vae.decoderConv6, nil, []int{64}},
		{"decoder_conv.7.weight", vae.decoderConv7, []int{64}, nil},
		{"decoder_conv.7.bias", vae.decoderConv7, nil, []int{64}},
		{"decoder_conv.7.weight", vae.decoderConv7, []int{64}, nil}, // Optional
		{"decoder_conv.7.bias", vae.decoderConv7, []int{64}, nil},   // Optional
		{"decoder_conv.9.weight", vae.decoderConv9, []int{64, 3, 5, 5}, nil},
		{"decoder_conv.9.bias", vae.decoderConv9, nil, []int{3}},
	}
	var weightsDir string
	{
		d, err := os.Getwd()
		if err != nil {
			panic(fmt.Sprint("Error getting working directory: %v\n", err))
		}
		// Assuming weights are in a subdirectory structure like project_root/py/data/
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
		} else if loadInfos[i].biasShape != nil {
			biasFilePath := filepath.Join(weightsDir, loadInfos[i].pyTorchName+".csv")
			data, err := torch.LoadFlatDataFromCSV(biasFilePath)
			if err != nil {
				panic(err)
			}
			loadInfos[i].goLayer.SetBiasAndShape(data, loadInfos[i].biasShape)
		}
	}

	fmt.Println("\n--- VAE model parameters loaded successfully. ---")
	return vae
}

func main() {

	vae := NewVAE()
	fmt.Println(vae)
	fmt.Println("VAE model created and loaded.")
}
