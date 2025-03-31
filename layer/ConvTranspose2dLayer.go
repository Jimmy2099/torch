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
		panic(fmt.Sprintf("Weight tensor must be 4D (outChannels, inChannels, kernelSizeRows, kernelSizeCols), but got %v", ct.Weights.Shape))
	}

	outChannels := ct.Weights.Shape[0]
	kernelSizeRows := ct.KernelSizeRow
	kernelSizeCols := ct.KernelSizeCol

	// Check consistency (use stored InChannels from constructor)
	if inChannels != ct.InChannels {
		panic(fmt.Sprintf("Input channels (%d) must match layer's expected inChannels (%d)", inChannels, ct.InChannels))
	}
	if ct.Weights.Shape[1] != ct.InChannels {
		panic(fmt.Sprintf("Weight's inChannels dimension (%d) must match layer's expected inChannels (%d)", ct.Weights.Shape[1], ct.InChannels))
	}

	// Calculate output dimensions (This formula seems correct)
	outputHeight := (inputHeight-1)*ct.StrideRow - 2*ct.PaddingRow + ct.KernelSizeRow + ct.OutputPaddingRow
	outputWidth := (inputWidth-1)*ct.StrideCol - 2*ct.PaddingCol + ct.KernelSizeCol + ct.OutputPaddingCol

	// --- Defensive check for non-positive output dimensions ---
	if outputHeight <= 0 || outputWidth <= 0 {
		panic(fmt.Sprintf("Calculated output dimensions are not positive: H=%d, W=%d. Check parameters (stride, padding, kernel size, input size).", outputHeight, outputWidth))
	}
	// --- End Defensive check ---

	// Create output tensor
	outputData := make([]float64, batchSize*outChannels*outputHeight*outputWidth)
	output := tensor.NewTensor(outputData, []int{batchSize, outChannels, outputHeight, outputWidth})

	// Perform transposed convolution
	for b := 0; b < batchSize; b++ {
		for outC := 0; outC < outChannels; outC++ {
			for i := 0; i < outputHeight; i++ { // Loop over output height
				for j := 0; j < outputWidth; j++ { // Loop over output width
					sum := 0.0
					// Iterate through the kernel
					for kRow := 0; kRow < kernelSizeRows; kRow++ {
						for kCol := 0; kCol < kernelSizeCols; kCol++ {

							// --- CORRECTED MAPPING LOGIC ---
							numeratorRow := i + ct.PaddingRow - kRow
							numeratorCol := j + ct.PaddingCol - kCol

							// Check if this kernel position maps to a valid potential input location via stride
							if numeratorRow >= 0 && numeratorRow%ct.StrideRow == 0 &&
								numeratorCol >= 0 && numeratorCol%ct.StrideCol == 0 {

								inputRow := numeratorRow / ct.StrideRow
								inputCol := numeratorCol / ct.StrideCol

								// Check if the mapped input location is within the actual input bounds
								if inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth {
									// If valid, accumulate contributions from all input channels
									for inC := 0; inC < inChannels; inC++ {
										// Weight index: Corresponds to W[outC, inC, kRow, kCol]
										weightIndex := outC*inChannels*kernelSizeRows*kernelSizeCols + // Offset for output channel
											inC*kernelSizeRows*kernelSizeCols + // Offset for input channel
											kRow*kernelSizeCols + // Offset for kernel row
											kCol // Offset for kernel col

										// Input index: Corresponds to Input[b, inC, inputRow, inputCol]
										inputIndex := b*inChannels*inputHeight*inputWidth + // Offset for batch
											inC*inputHeight*inputWidth + // Offset for input channel
											inputRow*inputWidth + // Offset for input row
											inputCol // Offset for input col

										// --- Bounds check before access (Defensive) ---
										if inputIndex < 0 || inputIndex >= len(inputTensor.Data) {
											panic(fmt.Sprintf("Internal error: Calculated inputIndex %d is out of bounds [0, %d). Params: b=%d, inC=%d, iR=%d, iC=%d, i=%d, j=%d, kR=%d, kC=%d",
												inputIndex, len(inputTensor.Data), b, inC, inputRow, inputCol, i, j, kRow, kCol))
										}
										if weightIndex < 0 || weightIndex >= len(ct.Weights.Data) {
											panic(fmt.Sprintf("Internal error: Calculated weightIndex %d is out of bounds [0, %d). Params: outC=%d, inC=%d, kR=%d, kC=%d",
												weightIndex, len(ct.Weights.Data), outC, inC, kRow, kCol))
										}
										// --- End Bounds check ---

										sum += inputTensor.Data[inputIndex] * ct.Weights.Data[weightIndex]
									} // end loop over inC
								} // end check: inputRow/Col in bounds
							} // end check: divisibility by stride
							// --- END CORRECTED MAPPING LOGIC ---

						} // end loop kCol
					} // end loop kRow

					// Add bias for the current output channel
					outputIndex := b*outChannels*outputHeight*outputWidth + // Offset for batch
						outC*outputHeight*outputWidth + // Offset for output channel
						i*outputWidth + // Offset for output row
						j // Offset for output col

					// --- Bounds check before access (Defensive) ---
					if outputIndex < 0 || outputIndex >= len(output.Data) {
						panic(fmt.Sprintf("Internal error: Calculated outputIndex %d is out of bounds [0, %d). Params: b=%d, outC=%d, i=%d, j=%d",
							outputIndex, len(output.Data), b, outC, i, j))
					}
					if outC < 0 || outC >= len(ct.Bias.Data) {
						panic(fmt.Sprintf("Internal error: Calculated bias index %d is out of bounds [0, %d).", outC, len(ct.Bias.Data)))
					}
					// --- End Bounds check ---

					output.Data[outputIndex] = sum + ct.Bias.Data[outC]
				} // end loop j (output width)
			} // end loop i (output height)
		} // end loop outC
	} // end loop b (batch)

	// fmt.Println("ConvTranspose2dLayer Forward Pass Completed") // Keep if useful
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
