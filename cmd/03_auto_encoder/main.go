// Filename: auto_encoder.go
package main

import (
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"os"
	"path/filepath"
)

// AutoEncoder defines the structure of the autoencoder model.
// Based on the Python structure:
// Autoencoder(
//
//	(encoder): Sequential(
//	  (0): Linear(in_features=784, out_features=128, bias=True) -> fc1
//	  (1): ReLU()                                            -> relu1
//	  (2): Linear(in_features=128, out_features=64, bias=True)  -> fc2
//	  (3): ReLU()                                            -> relu2
//	  (4): Linear(in_features=64, out_features=32, bias=True)   -> fc3
//	)
//	(decoder): Sequential(
//	  (0): Linear(in_features=32, out_features=64, bias=True)   -> fc4
//	  (1): ReLU()                                            -> relu3
//	  (2): Linear(in_features=64, out_features=128, bias=True)  -> fc5
//	  (3): ReLU()                                            -> relu4
//	  (4): Linear(in_features=128, out_features=784, bias=True) -> fc6
//	  (5): Sigmoid()                                         -> Sigmoid1
//	)
//
// )
type AutoEncoder struct {
	// Encoder layers
	fc1   *torch.LinearLayer
	relu1 *torch.ReLULayer
	fc2   *torch.LinearLayer
	relu2 *torch.ReLULayer
	fc3   *torch.LinearLayer // Latent space representation

	// Decoder layers
	fc4      *torch.LinearLayer
	relu3    *torch.ReLULayer
	fc5      *torch.LinearLayer
	relu4    *torch.ReLULayer
	fc6      *torch.LinearLayer
	Sigmoid1 *torch.SigmoidLayer // Output activation
}

// NewAutoEncoder creates and initializes a new AutoEncoder model.
// It also loads the pre-trained weights and biases from CSV files.
func NewAutoEncoder() *AutoEncoder {
	// 1. Initialize the AutoEncoder struct with new layers
	ae := &AutoEncoder{
		// Encoder
		fc1:   torch.NewLinearLayer(784, 128),
		relu1: torch.NewReLULayer(),
		fc2:   torch.NewLinearLayer(128, 64),
		relu2: torch.NewReLULayer(),
		fc3:   torch.NewLinearLayer(64, 32),

		// Decoder
		fc4:      torch.NewLinearLayer(32, 64),
		relu3:    torch.NewReLULayer(),
		fc5:      torch.NewLinearLayer(64, 128),
		relu4:    torch.NewReLULayer(),
		fc6:      torch.NewLinearLayer(128, 784),
		Sigmoid1: torch.NewSigmoidLayer(),
	}

	// 2. Define layer file name components and shapes
	layerFileNames := []string{
		"encoder.0", // Corresponds to ae.fc1
		"encoder.2", // Corresponds to ae.fc2
		"encoder.4", // Corresponds to ae.fc3
		"decoder.0", // Corresponds to ae.fc4
		"decoder.2", // Corresponds to ae.fc5
		"decoder.4", // Corresponds to ae.fc6
	}

	weightShapes := [][]int{
		{128, 784}, // fc1
		{64, 128},  // fc2
		{32, 64},   // fc3
		{64, 32},   // fc4
		{128, 64},  // fc5
		{784, 128}, // fc6
	}
	biasShapes := [][]int{
		{128}, // fc1
		{64},  // fc2
		{32},  // fc3
		{64},  // fc4
		{128}, // fc5
		{784}, // fc6
	}

	// 3. Get base data path
	basePath := "py/data" // Relative path to data directory
	d, err := os.Getwd()
	if err != nil {
		panic(fmt.Sprintf("Error getting working directory: %v", err))
	}
	dataPath := filepath.Join(d, basePath)

	fmt.Println("Loading AutoEncoder model parameters...")

	// 4. Load parameters for each layer in separate blocks

	// --- Layer fc1 (encoder.0) ---
	{
		layerIdx := 0
		layerName := layerFileNames[layerIdx]
		wShape := weightShapes[layerIdx]
		bShape := biasShapes[layerIdx]
		fmt.Printf("--- Loading %s ---\n", layerName)

		// Load Weights
		weightFilePath := filepath.Join(dataPath, layerName+".weight.csv")
		weightData, err := torch.LoadMatrixFromCSV(weightFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading weight file %s: %v", weightFilePath, err))
		}
		ae.fc1.SetWeights(weightData.Data) // Set weights for fc1
		ae.fc1.Weights.Reshape(wShape)     // Reshape weights for fc1
		fmt.Printf("Loaded weights for %s, shape: %v\n", layerName, ae.fc1.Weights.Shape)

		// Load Biases
		biasFilePath := filepath.Join(dataPath, layerName+".bias.csv")
		biasData, err := torch.LoadMatrixFromCSV(biasFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading bias file %s: %v", biasFilePath, err))
		}
		ae.fc1.SetBias(biasData.Data) // Set bias for fc1
		ae.fc1.Bias.Reshape(bShape)   // Reshape bias for fc1
		fmt.Printf("Loaded biases for %s, shape: %v\n", layerName, ae.fc1.Bias.Shape)
	}

	// --- Layer fc2 (encoder.2) ---
	{
		layerIdx := 1
		layerName := layerFileNames[layerIdx]
		wShape := weightShapes[layerIdx]
		bShape := biasShapes[layerIdx]
		fmt.Printf("--- Loading %s ---\n", layerName)

		// Load Weights
		weightFilePath := filepath.Join(dataPath, layerName+".weight.csv")
		weightData, err := torch.LoadMatrixFromCSV(weightFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading weight file %s: %v", weightFilePath, err))
		}
		ae.fc2.SetWeights(weightData.Data) // Set weights for fc2
		ae.fc2.Weights.Reshape(wShape)     // Reshape weights for fc2
		fmt.Printf("Loaded weights for %s, shape: %v\n", layerName, ae.fc2.Weights.Shape)

		// Load Biases
		biasFilePath := filepath.Join(dataPath, layerName+".bias.csv")
		biasData, err := torch.LoadMatrixFromCSV(biasFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading bias file %s: %v", biasFilePath, err))
		}
		ae.fc2.SetBias(biasData.Data) // Set bias for fc2
		ae.fc2.Bias.Reshape(bShape)   // Reshape bias for fc2
		fmt.Printf("Loaded biases for %s, shape: %v\n", layerName, ae.fc2.Bias.Shape)
	}

	// --- Layer fc3 (encoder.4) ---
	{
		layerIdx := 2
		layerName := layerFileNames[layerIdx]
		wShape := weightShapes[layerIdx]
		bShape := biasShapes[layerIdx]
		fmt.Printf("--- Loading %s ---\n", layerName)

		// Load Weights
		weightFilePath := filepath.Join(dataPath, layerName+".weight.csv")
		weightData, err := torch.LoadMatrixFromCSV(weightFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading weight file %s: %v", weightFilePath, err))
		}
		ae.fc3.SetWeights(weightData.Data) // Set weights for fc3
		ae.fc3.Weights.Reshape(wShape)     // Reshape weights for fc3
		fmt.Printf("Loaded weights for %s, shape: %v\n", layerName, ae.fc3.Weights.Shape)

		// Load Biases
		biasFilePath := filepath.Join(dataPath, layerName+".bias.csv")
		biasData, err := torch.LoadMatrixFromCSV(biasFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading bias file %s: %v", biasFilePath, err))
		}
		ae.fc3.SetBias(biasData.Data) // Set bias for fc3
		ae.fc3.Bias.Reshape(bShape)   // Reshape bias for fc3
		fmt.Printf("Loaded biases for %s, shape: %v\n", layerName, ae.fc3.Bias.Shape)
	}

	// --- Layer fc4 (decoder.0) ---
	{
		layerIdx := 3
		layerName := layerFileNames[layerIdx]
		wShape := weightShapes[layerIdx]
		bShape := biasShapes[layerIdx]
		fmt.Printf("--- Loading %s ---\n", layerName)

		// Load Weights
		weightFilePath := filepath.Join(dataPath, layerName+".weight.csv")
		weightData, err := torch.LoadMatrixFromCSV(weightFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading weight file %s: %v", weightFilePath, err))
		}
		ae.fc4.SetWeights(weightData.Data) // Set weights for fc4
		ae.fc4.Weights.Reshape(wShape)     // Reshape weights for fc4
		fmt.Printf("Loaded weights for %s, shape: %v\n", layerName, ae.fc4.Weights.Shape)

		// Load Biases
		biasFilePath := filepath.Join(dataPath, layerName+".bias.csv")
		biasData, err := torch.LoadMatrixFromCSV(biasFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading bias file %s: %v", biasFilePath, err))
		}
		ae.fc4.SetBias(biasData.Data) // Set bias for fc4
		ae.fc4.Bias.Reshape(bShape)   // Reshape bias for fc4
		fmt.Printf("Loaded biases for %s, shape: %v\n", layerName, ae.fc4.Bias.Shape)
	}

	// --- Layer fc5 (decoder.2) ---
	{
		layerIdx := 4
		layerName := layerFileNames[layerIdx]
		wShape := weightShapes[layerIdx]
		bShape := biasShapes[layerIdx]
		fmt.Printf("--- Loading %s ---\n", layerName)

		// Load Weights
		weightFilePath := filepath.Join(dataPath, layerName+".weight.csv")
		weightData, err := torch.LoadMatrixFromCSV(weightFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading weight file %s: %v", weightFilePath, err))
		}
		ae.fc5.SetWeights(weightData.Data) // Set weights for fc5
		ae.fc5.Weights.Reshape(wShape)     // Reshape weights for fc5
		fmt.Printf("Loaded weights for %s, shape: %v\n", layerName, ae.fc5.Weights.Shape)

		// Load Biases
		biasFilePath := filepath.Join(dataPath, layerName+".bias.csv")
		biasData, err := torch.LoadMatrixFromCSV(biasFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading bias file %s: %v", biasFilePath, err))
		}
		ae.fc5.SetBias(biasData.Data) // Set bias for fc5
		ae.fc5.Bias.Reshape(bShape)   // Reshape bias for fc5
		fmt.Printf("Loaded biases for %s, shape: %v\n", layerName, ae.fc5.Bias.Shape)
	}

	// --- Layer fc6 (decoder.4) ---
	{
		layerIdx := 5
		layerName := layerFileNames[layerIdx]
		wShape := weightShapes[layerIdx]
		bShape := biasShapes[layerIdx]
		fmt.Printf("--- Loading %s ---\n", layerName)

		// Load Weights
		weightFilePath := filepath.Join(dataPath, layerName+".weight.csv")
		weightData, err := torch.LoadMatrixFromCSV(weightFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading weight file %s: %v", weightFilePath, err))
		}
		ae.fc6.SetWeights(weightData.Data) // Set weights for fc6
		ae.fc6.Weights.Reshape(wShape)     // Reshape weights for fc6
		fmt.Printf("Loaded weights for %s, shape: %v\n", layerName, ae.fc6.Weights.Shape)

		// Load Biases
		biasFilePath := filepath.Join(dataPath, layerName+".bias.csv")
		biasData, err := torch.LoadMatrixFromCSV(biasFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading bias file %s: %v", biasFilePath, err))
		}
		ae.fc6.SetBias(biasData.Data) // Set bias for fc6
		ae.fc6.Bias.Reshape(bShape)   // Reshape bias for fc6
		fmt.Printf("Loaded biases for %s, shape: %v\n", layerName, ae.fc6.Bias.Shape)
	}

	fmt.Println("--- AutoEncoder model parameters loaded successfully. ---")

	// 5. Return the initialized model
	return ae
}

// loadWeightsAndBiasesAE loads the weights and biases for the AutoEncoder from CSV files.
func loadWeightsAndBiasesAE(ae *AutoEncoder) {
	// Define the layer names as expected in the file system (e.g., fc1, fc2, ...)
	// These should correspond to how the weights were saved (e.g., from Python).
	layerNames := []string{"fc1", "fc2", "fc3", "fc4", "fc5", "fc6"}
	layers := []*torch.LinearLayer{ae.fc1, ae.fc2, ae.fc3, ae.fc4, ae.fc5, ae.fc6}
	// Expected shapes for weights [out_features, in_features] and biases [out_features]
	// Note: LinearLayer weights are often stored/used as [out_features, in_features]
	weightShapes := [][]int{
		{128, 784}, // fc1
		{64, 128},  // fc2
		{32, 64},   // fc3
		{64, 32},   // fc4
		{128, 64},  // fc5
		{784, 128}, // fc6
	}
	biasShapes := [][]int{
		{128}, // fc1
		{64},  // fc2
		{32},  // fc3
		{64},  // fc4
		{128}, // fc5
		{784}, // fc6
	}

	basePath := "py/data" // Assuming data files are in 'py/data' relative to execution directory
	d, err := os.Getwd()
	if err != nil {
		panic(fmt.Sprintf("Error getting working directory: %v", err))
	}
	dataPath := filepath.Join(d, basePath)

	fmt.Println("Loading AutoEncoder model parameters...")

	for i, layerName := range layerNames {
		layer := layers[i]
		wShape := weightShapes[i]
		bShape := biasShapes[i]

		// Load Weights
		weightFilePath := filepath.Join(dataPath, layerName+".weight.csv")
		fmt.Printf("Loading weights for %s from %s\n", layerName, weightFilePath)
		weightData, err := torch.LoadMatrixFromCSV(weightFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading weight file %s: %v", weightFilePath, err))
		}
		layer.SetWeights(weightData.Data) // SetWeights likely expects [][]float64
		// Reshape the internal tensor representation
		layer.Weights = layer.Weights.Reshape(wShape)

		fmt.Printf("Loaded weights for %s, shape: %v\n", layerName, layer.Weights.Shape)

		// Load Biases
		biasFilePath := filepath.Join(dataPath, layerName+".bias.csv")
		fmt.Printf("Loading biases for %s from %s\n", layerName, biasFilePath)
		biasData, err := torch.LoadMatrixFromCSV(biasFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading bias file %s: %v", biasFilePath, err))
		}
		layer.SetBias(biasData.Data) // SetBias likely expects [][]float64
		// Reshape the internal tensor representation
		layer.Bias = layer.Bias.Reshape(bShape)

		fmt.Printf("Loaded biases for %s, shape: %v\n", layerName, layer.Bias.Shape)
		fmt.Println("---")
	}
	fmt.Println("AutoEncoder model parameters loaded successfully.")
}

// Forward performs the forward pass of the AutoEncoder.
func (ae *AutoEncoder) Forward(x *tensor.Tensor) *tensor.Tensor {
	// Input should be flattened image, e.g., shape [batch_size, 784] or [1, 784]
	if len(x.Shape) != 2 || x.Shape[1] != 784 {
		// Attempt to flatten if not already flat (e.g., if input is [1, 1, 28, 28])
		if x.Size() == 784 {
			fmt.Printf("Input shape %v is not [N, 784], attempting to flatten.\n", x.Shape)
			x = x.Flatten() // Flatten to [1, 784]
			fmt.Printf("Flattened input shape: %v\n", x.Shape)
		} else {
			panic(fmt.Sprintf("Input tensor shape %v is incompatible with AutoEncoder input (expected [N, 784])", x.Shape))
		}
	}

	fmt.Println("\n=== Starting AutoEncoder Forward Pass ===")
	fmt.Printf("Input shape: %v\n", x.Shape)

	// --- Encoder ---
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

	fmt.Println("\nEncoder FC3 (Latent Space):")
	x = ae.fc3.Forward(x)
	fmt.Printf("After fc3 (latent): %v\n", x.Shape)

	// --- Decoder ---
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

	fmt.Println("\nSigmoid (Output):")
	x = ae.Sigmoid1.Forward(x)
	fmt.Printf("After sigmoid (output): %v\n", x.Shape) // Should be [N, 784]

	fmt.Println("\n=== AutoEncoder Forward Pass Complete ===")
	return x
}

// Parameters returns a slice of all trainable parameters (weights and biases) in the model.
// Note: Returning *tensor.Tensor based on LinearLayer structure.
func (ae *AutoEncoder) Parameters() []*tensor.Tensor {
	params := make([]*tensor.Tensor, 0)
	// Encoder parameters
	params = append(params, ae.fc1.Weights, ae.fc1.Bias)
	params = append(params, ae.fc2.Weights, ae.fc2.Bias)
	params = append(params, ae.fc3.Weights, ae.fc3.Bias)
	// Decoder parameters
	params = append(params, ae.fc4.Weights, ae.fc4.Bias)
	params = append(params, ae.fc5.Weights, ae.fc5.Bias)
	params = append(params, ae.fc6.Weights, ae.fc6.Bias)
	return params
}

// ZeroGrad resets the gradients of all trainable parameters in the model.
func (ae *AutoEncoder) ZeroGrad() {
	ae.fc1.ZeroGrad()
	ae.fc2.ZeroGrad()
	ae.fc3.ZeroGrad()
	ae.fc4.ZeroGrad()
	ae.fc5.ZeroGrad()
	ae.fc6.ZeroGrad()
	// ReLU and Sigmoid layers typically don't have trainable parameters or gradients to zero.
}

func main() {
	// Create AutoEncoder model (this will also load weights/biases)
	fmt.Println("Creating AutoEncoder model...")
	model := NewAutoEncoder()
	fmt.Println("AutoEncoder model created.")

	// Example input (batch size 1, flattened 28x28 image = 784 features)
	// Create a dummy tensor with shape [1, 784]
	// In a real scenario, you would load an actual image and flatten it.
	inputData := make([]float64, 784)
	// Fill with some dummy values (e.g., 0.5)
	for i := range inputData {
		inputData[i] = 0.5
	}
	input := tensor.NewTensor(inputData, []int{1, 784})

	fmt.Printf("\nCreated dummy input tensor with shape: %v\n", input.Shape)

	// Perform the forward pass
	fmt.Println("\nStarting forward pass with dummy input...")
	output := model.Forward(input)
	fmt.Println("\nForward pass finished.")

	// Print the output shape (should be the same as input shape: [1, 784])
	fmt.Printf("Output tensor shape: %v\n", output.Shape)

	// Optionally, print some output values (e.g., first 10)
	fmt.Println("First 10 output values:", output.Data[:10])

	// You could also test the Parameters() and ZeroGrad() methods
	// params := model.Parameters()
	// fmt.Printf("\nNumber of parameter tensors: %d\n", len(params))
	// model.ZeroGrad()
	// fmt.Println("Gradients zeroed (notional test).")
}
