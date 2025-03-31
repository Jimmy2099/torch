package layer

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"math"
	"math/rand"
)

// ConvTranspose2dLayer implements the transposed convolution operation.
type ConvTranspose2dLayer struct {
	Weights          *tensor.Tensor
	Bias             *tensor.Tensor
	KernelSizeRow    int
	KernelSizeCol    int
	StrideRow        int
	StrideCol        int
	PaddingRow       int
	PaddingCol       int
	OutputPaddingRow int
	OutputPaddingCol int
	InChannels       int
	inputCache       *tensor.Tensor // Cache for input tensor needed in backward pass
}

// NewConvTranspose2dLayer creates a new ConvTranspose2dLayer.
// inChannels: Number of input channels
// outChannels: Number of output channels
// kernelSize: Size of the convolutional kernel (kernelSize x kernelSize)
// stride: Stride of the convolution
// padding: Padding applied to the input
// outputPadding: Additional size added to one side of the output shape
func NewConvTranspose2dLayer(inChannels, outChannels, kernelSizeRows, kernelSizeCols, strideRows, strideCols, paddingRows, paddingCols, outputPaddingRows, outputPaddingCols int) *ConvTranspose2dLayer {
	// Initialize weights with Kaiming He initialization
	weightsData := make([]float64, inChannels*outChannels*kernelSizeRows*kernelSizeCols)
	variance := 2.0 / float64(inChannels*kernelSizeRows*kernelSizeCols)
	for i := range weightsData {
		weightsData[i] = randn() * math.Sqrt(variance)
	}

	weights := tensor.NewTensor(weightsData, []int{outChannels, inChannels, kernelSizeRows, kernelSizeCols}) // Weight shape adjusted
	// Initialize bias to zero
	bias := tensor.NewTensor(make([]float64, outChannels), []int{outChannels})

	return &ConvTranspose2dLayer{
		Weights:          weights,
		Bias:             bias,
		KernelSizeRow:    kernelSizeRows,
		KernelSizeCol:    kernelSizeCols,
		StrideRow:        strideRows,
		StrideCol:        strideCols,
		PaddingRow:       paddingRows,
		PaddingCol:       paddingCols,
		OutputPaddingRow: outputPaddingRows,
		OutputPaddingCol: outputPaddingCols,
		InChannels:       inChannels, // Store inChannels
		inputCache:       nil,        // Initialize cache to nil
	}
}

// randn generates a random number with standard normal distribution.
func randn() float64 {
	const (
		twoPi = 2 * math.Pi
	)
	var (
		u1, u2, r float64
	)
	for {
		u1 = rand.Float64()
		u2 = rand.Float64()
		r = math.Sqrt(-2 * math.Log(u1) * math.Cos(twoPi*u2))
		if !math.IsNaN(r) {
			break
		}
	}
	return r
}

// Forward performs the forward pass through the transposed convolutional layer.
// input: Input tensor of shape [batchSize, inChannels, inputHeight, inputWidth]
// Returns: Output tensor of shape [batchSize, outChannels, outputHeight, outputWidth]
func (ct *ConvTranspose2dLayer) Forward(inputTensor *tensor.Tensor) *tensor.Tensor {
	// Check input dimensions
	if len(inputTensor.Shape) != 4 {
		panic(fmt.Sprintf("Input tensor must be 4D (batchSize, inChannels, height, width), but got %v", inputTensor.Shape))
	}

	// Store input for backward pass
	ct.inputCache = inputTensor

	batchSize := inputTensor.Shape[0]
	inChannels := inputTensor.Shape[1]
	inputHeight := inputTensor.Shape[2]
	inputWidth := inputTensor.Shape[3]

	// Check weight dimensions
	if len(ct.Weights.Shape) != 4 {
		panic(fmt.Sprintf("Weight tensor must be 4D (outChannels, inChannels, kernelSize, kernelSize), but got %v", ct.Weights.Shape))
	}

	outChannels := ct.Weights.Shape[0]
	//weightInChannels := ct.Weights.Shape[1] // This is correct now
	kernelSizeRows := ct.KernelSizeRow
	kernelSizeCols := ct.KernelSizeCol

	if inChannels != ct.InChannels { // Use stored inChannels
		panic(fmt.Sprintf("Input channels (%d) must match weight input channels (%d)", inChannels, ct.InChannels))
	}

	// Calculate output dimensions
	outputHeight := (inputHeight-1)*ct.StrideRow - 2*ct.PaddingRow + ct.KernelSizeRow + ct.OutputPaddingRow
	outputWidth := (inputWidth-1)*ct.StrideCol - 2*ct.PaddingCol + ct.KernelSizeCol + ct.OutputPaddingCol

	// Create output tensor
	outputData := make([]float64, batchSize*outChannels*outputHeight*outputWidth)
	output := tensor.NewTensor(outputData, []int{batchSize, outChannels, outputHeight, outputWidth})

	// Perform transposed convolution
	for b := 0; b < batchSize; b++ {
		for outC := 0; outC < outChannels; outC++ {
			for i := 0; i < outputHeight; i++ {
				for j := 0; j < outputWidth; j++ {
					//Accumulate the results from the convolution
					sum := 0.0
					//Cycle through the locations in the kernel
					for kRow := 0; kRow < kernelSizeRows; kRow++ {
						for kCol := 0; kCol < kernelSizeCols; kCol++ {
							//Get the current location in the original image based on location in the output and kernel
							inputRow := (i + 2*ct.PaddingRow - kRow - ct.OutputPaddingRow) / ct.StrideRow
							inputCol := (j + 2*ct.PaddingCol - kCol - ct.OutputPaddingCol) / ct.StrideCol

							// If row and column are within the range of the input then use that location
							if inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth {
								for inC := 0; inC < inChannels; inC++ {

									//Determine the weight index
									weightIndex := outC*ct.InChannels*kernelSizeRows*kernelSizeCols + inC*kernelSizeRows*kernelSizeCols + kRow*kernelSizeCols + kCol

									//Input index
									inputIndex := b*inChannels*inputHeight*inputWidth + inC*inputHeight*inputWidth + inputRow*inputWidth + inputCol

									sum += inputTensor.Data[inputIndex] * ct.Weights.Data[weightIndex]

								}

							}

						}
					}
					//Add the calculated bias and sum for the output location
					outputIndex := b*outChannels*outputHeight*outputWidth + outC*outputHeight*outputWidth + i*outputWidth + j
					output.Data[outputIndex] = sum + ct.Bias.Data[outC]
				}
			}
		}
	}
	fmt.Println("ConvTranspose2dLayer Forward Pass")
	return output
}

// Backward performs the backward pass through the transposed convolutional layer (gradient calculation).
// gradOutput: Gradient of the loss with respect to the output of this layer.
// learningRate: Learning rate for updating weights and biases.
// Returns: Gradient of the loss with respect to the input of this layer.
func (ct *ConvTranspose2dLayer) Backward(gradOutput *tensor.Tensor, learningRate float64) *tensor.Tensor {
	// Ensure input tensor was cached during forward pass
	if ct.inputCache == nil {
		panic("Backward called before Forward or input cache is nil")
	}
	inputTensor := ct.inputCache // Use the cached input tensor

	// Check gradOutput dimensions
	if len(gradOutput.Shape) != 4 {
		panic(fmt.Sprintf("gradOutput tensor must be 4D, but got %v", gradOutput.Shape))
	}
	batchSize := gradOutput.Shape[0]
	outChannels := gradOutput.Shape[1]
	outputHeight := gradOutput.Shape[2]
	outputWidth := gradOutput.Shape[3]

	// Input tensor (for calculating gradInput)
	inChannels := ct.Weights.Shape[1]
	//Get the kernel information
	kernelSizeRows := ct.KernelSizeRow
	kernelSizeCols := ct.KernelSizeCol

	//Calculate the height and width of the input
	inputHeight := (outputHeight+2*ct.PaddingRow-ct.KernelSizeRow-ct.OutputPaddingRow)/ct.StrideRow + 1
	inputWidth := (outputWidth+2*ct.PaddingCol-ct.KernelSizeCol-ct.OutputPaddingCol)/ct.StrideCol + 1

	// Initialize gradInput
	gradInputData := make([]float64, batchSize*inChannels*inputHeight*inputWidth)
	gradInput := tensor.NewTensor(gradInputData, []int{batchSize, inChannels, inputHeight, inputWidth})

	// Initialize gradients for weights and biases (you might want to store these in the ConvTranspose2dLayer struct if training)
	gradWeightsData := make([]float64, len(ct.Weights.Data))
	gradWeights := tensor.NewTensor(gradWeightsData, ct.Weights.Shape) // Assuming you have the same shape as the weights
	gradBiasData := make([]float64, len(ct.Bias.Data))
	gradBias := tensor.NewTensor(gradBiasData, ct.Bias.Shape) // Assuming the same shape as bias

	// Iterate through output to calculate gradients
	for b := 0; b < batchSize; b++ {
		for outC := 0; outC < outChannels; outC++ {
			for i := 0; i < outputHeight; i++ {
				for j := 0; j < outputWidth; j++ {
					outputIndex := b*outChannels*outputHeight*outputWidth + outC*outputHeight*outputWidth + i*outputWidth + j
					gradOutputValue := gradOutput.Data[outputIndex]

					//Cycle through the locations in the kernel
					for kRow := 0; kRow < kernelSizeRows; kRow++ {
						for kCol := 0; kCol < kernelSizeCols; kCol++ {
							//Get the current location in the original image based on location in the output and kernel
							inputRow := (i + 2*ct.PaddingRow - kRow - ct.OutputPaddingRow) / ct.StrideRow
							inputCol := (j + 2*ct.PaddingCol - kCol - ct.OutputPaddingCol) / ct.StrideCol

							// If row and column are within the range of the input then use that location
							if inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth {
								for inC := 0; inC < inChannels; inC++ {
									//Input index
									inputIndex := b*inChannels*inputHeight*inputWidth + inC*inputHeight*inputWidth + inputRow*inputWidth + inputCol

									//Determine the weight index
									weightIndex := outC*ct.InChannels*kernelSizeRows*kernelSizeCols + inC*kernelSizeRows*kernelSizeCols + kRow*kernelSizeCols + kCol
									//Calculate gradient from output back to input location based on outputIndex,inputIndex, and kernels
									gradInput.Data[inputIndex] += gradOutputValue * ct.Weights.Data[weightIndex]
									// Update weight gradient using the cached input tensor
									gradWeights.Data[weightIndex] += gradOutputValue * inputTensor.Data[inputIndex] // Corrected line

								}

							}

						}
					}
					gradBias.Data[outC] += gradOutputValue

				}
			}
		}
	}

	// Update weights and biases using the calculated gradients
	for i := range ct.Weights.Data {
		ct.Weights.Data[i] -= learningRate * gradWeights.Data[i]
	}
	for i := range ct.Bias.Data {
		ct.Bias.Data[i] -= learningRate * gradBias.Data[i]
	}

	//Zero the data to the gradients (Optional: gradients are usually zeroed by the optimizer)
	//for i := range gradWeights.Data {
	//	gradWeights.Data[i] = 0
	//}
	//for i := range gradBias.Data {
	//	gradBias.Data[i] = 0
	//}

	return gradInput
}

// ZeroGrad resets the gradients of weights and biases to zero.
// Note: Typically, the optimizer handles zeroing gradients.
// This method might be redundant if using a standard optimizer loop.
// However, implementing it for completeness is fine.
func (ct *ConvTranspose2dLayer) ZeroGrad() {
	// We don't explicitly store gradients in the layer anymore for this example,
	// as they are calculated and used within the Backward method.
	// If you were to store gradWeights and gradBias in the struct,
	// you would zero them here.
}

// Parameters returns a slice containing the weights and bias tensors of the layer.
func (ct *ConvTranspose2dLayer) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{ct.Weights, ct.Bias}
}

// SetWeights sets the weights of the layer.
func (ct *ConvTranspose2dLayer) SetWeights(data [][]float64) {
	// Convert the 2D slice to a 1D slice
	flattenedData := make([]float64, 0)
	for _, row := range data {
		flattenedData = append(flattenedData, row...)
	}

	// Create a new tensor with the flattened data and the original shape
	ct.Weights = tensor.NewTensor(flattenedData, ct.Weights.Shape)

}

// SetBias sets the bias of the layer.
func (ct *ConvTranspose2dLayer) SetBias(data [][]float64) {
	// Convert the 2D slice to a 1D slice
	flattenedData := make([]float64, 0)
	for _, row := range data {
		flattenedData = append(flattenedData, row...)
	}
	ct.Bias = tensor.NewTensor(flattenedData, ct.Bias.Shape)
}

func (l *ConvTranspose2dLayer) SetWeightsAndShape(data []float64, shape []int) {
	l.Weights = tensor.NewTensor(data, shape)
}

func (l *ConvTranspose2dLayer) SetBiasAndShape(data []float64, shape []int) {
	l.Bias = tensor.NewTensor(data, shape)
}
