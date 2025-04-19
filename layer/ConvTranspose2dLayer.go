package layer

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"math/rand"
)

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
	OutChannels      int
	inputCache       *tensor.Tensor
}

func (ct *ConvTranspose2dLayer) GetWeights() *tensor.Tensor {
	return ct.Weights
}

func (ct *ConvTranspose2dLayer) GetBias() *tensor.Tensor {
	return ct.Bias
}

func NewConvTranspose2dLayer(inChannels, outChannels, kernelSizeRows, kernelSizeCols, strideRows, strideCols, paddingRows, paddingCols, outputPaddingRows, outputPaddingCols int) *ConvTranspose2dLayer {
	weightsData := make([]float32, inChannels*outChannels*kernelSizeRows*kernelSizeCols)

	fanIn := inChannels * kernelSizeRows * kernelSizeCols
	variance := 2.0 / float32(fanIn)
	for i := range weightsData {
		weightsData[i] = randn() * math.Sqrt(variance)
	}

	weights := tensor.NewTensor(weightsData, []int{inChannels, outChannels, kernelSizeRows, kernelSizeCols})
	bias := tensor.NewTensor(make([]float32, outChannels), []int{outChannels})

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
		InChannels:       inChannels,
		OutChannels:      outChannels,
		inputCache:       nil,
	}
}

func randn() float32 {
	const (
		twoPi = 2 * math.Pi
	)
	var (
		u1, u2, r float32
	)
	for {
		u1 = rand.Float32()
		u2 = rand.Float32()
		r = math.Sqrt(-2 * math.Log(u1) * math.Cos(twoPi*u2))
		if !math.IsNaN(r) {
			break
		}
	}
	return r
}

func (ct *ConvTranspose2dLayer) Forward(inputTensor *tensor.Tensor) *tensor.Tensor {
	if len(inputTensor.GetShape()) != 4 {
		panic(fmt.Sprintf("Input must be a 4D tensor [batch, in_channels, height, width], actual dimensions: %v", inputTensor.GetShape()))
	}
	batchSize := inputTensor.GetShapeByNum(0)
	inChannels := inputTensor.GetShapeByNum(1)
	inputHeight := inputTensor.GetShapeByNum(2)
	inputWidth := inputTensor.GetShapeByNum(3)

	if inChannels != ct.InChannels {
		panic(fmt.Sprintf("Input channel mismatch: Expected %d Actual %d", ct.InChannels, inChannels))
	}

	outputHeight := (inputHeight-1)*ct.StrideRow - 2*ct.PaddingRow + ct.KernelSizeRow + ct.OutputPaddingRow
	outputWidth := (inputWidth-1)*ct.StrideCol - 2*ct.PaddingCol + ct.KernelSizeCol + ct.OutputPaddingCol
	if outputHeight <= 0 || outputWidth <= 0 {
		panic(fmt.Sprintf("Illegal output size: %dx%d (parameters: stride=%dx%d padding=%dx%d kernel=%dx%d output_padding=%dx%d)",
			outputHeight, outputWidth, ct.StrideRow, ct.StrideCol, ct.PaddingRow, ct.PaddingCol,
			ct.KernelSizeRow, ct.KernelSizeCol, ct.OutputPaddingRow, ct.OutputPaddingCol))
	}

	outputShape := []int{batchSize, ct.OutChannels, outputHeight, outputWidth}
	outputData := make([]float32, product(outputShape))
	output := tensor.NewTensor(outputData, outputShape)

	for b := 0; b < batchSize; b++ {
		for outC := 0; outC < ct.OutChannels; outC++ {
			for i := 0; i < outputHeight; i++ {
				for j := 0; j < outputWidth; j++ {
					var sum float32

					for kRow := 0; kRow < ct.KernelSizeRow; kRow++ {
						for kCol := 0; kCol < ct.KernelSizeCol; kCol++ {
							numeratorRow := i + ct.PaddingRow - kRow
							numeratorCol := j + ct.PaddingCol - kCol

							if numeratorRow >= 0 && numeratorRow%ct.StrideRow == 0 &&
								numeratorCol >= 0 && numeratorCol%ct.StrideCol == 0 {

								inputRow := numeratorRow / ct.StrideRow
								inputCol := numeratorCol / ct.StrideCol

								if inputRow < inputHeight && inputCol < inputWidth {
									for inC := 0; inC < inChannels; inC++ {
										weightIndex := inC*ct.OutChannels*ct.KernelSizeRow*ct.KernelSizeCol +
											outC*ct.KernelSizeRow*ct.KernelSizeCol +
											kRow*ct.KernelSizeCol +
											kCol

										inputIndex := b*inChannels*inputHeight*inputWidth +
											inC*inputHeight*inputWidth +
											inputRow*inputWidth +
											inputCol

										if inputIndex < 0 || inputIndex >= len(inputTensor.Data) {
											panic(fmt.Sprintf("Input index out of bounds: %d (range 0-%d)", inputIndex, len(inputTensor.Data)))
										}
										if weightIndex < 0 || weightIndex >= len(ct.Weights.Data) {
											panic(fmt.Sprintf("Weight index out of bounds: %d (range 0-%d)", weightIndex, len(ct.Weights.Data)))
										}

										sum += inputTensor.Data[inputIndex] * ct.Weights.Data[weightIndex]
									}
								}
							}
						}
					}

					outputIndex := b*ct.OutChannels*outputHeight*outputWidth +
						outC*outputHeight*outputWidth +
						i*outputWidth +
						j

					if outC >= len(ct.Bias.Data) {
						panic(fmt.Sprintf("Bias index out of bounds: %d (total %d)", outC, len(ct.Bias.Data)))
					}
					output.Data[outputIndex] = sum + ct.Bias.Data[outC]
				}
			}
		}
	}

	ct.inputCache = inputTensor
	return output
}

func (ct *ConvTranspose2dLayer) Backward(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor {
	if ct.inputCache == nil {
		panic("Need to execute forward propagation first")
	}
	inputTensor := ct.inputCache

	batchSize := gradOutput.GetShape()[0]
	outChannels := gradOutput.GetShape()[1]
	outputHeight := gradOutput.GetShape()[2]
	outputWidth := gradOutput.GetShape()[3]

	inChannels := ct.InChannels
	inputHeight := inputTensor.GetShape()[2]
	inputWidth := inputTensor.GetShape()[3]

	gradInputShape := inputTensor.GetShape()
	gradInputData := make([]float32, product(gradInputShape))
	gradInput := tensor.NewTensor(gradInputData, gradInputShape)

	gradWeightsData := make([]float32, len(ct.Weights.Data))
	gradBiasData := make([]float32, len(ct.Bias.Data))

	for b := 0; b < batchSize; b++ {
		for outC := 0; outC < outChannels; outC++ {
			for i := 0; i < outputHeight; i++ {
				for j := 0; j < outputWidth; j++ {
					gradOutputIndex := b*outChannels*outputHeight*outputWidth +
						outC*outputHeight*outputWidth +
						i*outputWidth +
						j
					gradValue := gradOutput.Data[gradOutputIndex]

					gradBiasData[outC] += gradValue

					for kRow := 0; kRow < ct.KernelSizeRow; kRow++ {
						for kCol := 0; kCol < ct.KernelSizeCol; kCol++ {
							numeratorRow := i + ct.PaddingRow - kRow
							numeratorCol := j + ct.PaddingCol - kCol

							if numeratorRow >= 0 && numeratorRow%ct.StrideRow == 0 &&
								numeratorCol >= 0 && numeratorCol%ct.StrideCol == 0 {

								inputRow := numeratorRow / ct.StrideRow
								inputCol := numeratorCol / ct.StrideCol

								if inputRow < inputHeight && inputCol < inputWidth {
									for inC := 0; inC < inChannels; inC++ {
										weightIndex := inC*ct.OutChannels*ct.KernelSizeRow*ct.KernelSizeCol +
											outC*ct.KernelSizeRow*ct.KernelSizeCol +
											kRow*ct.KernelSizeCol +
											kCol

										inputIndex := b*inChannels*inputHeight*inputWidth +
											inC*inputHeight*inputWidth +
											inputRow*inputWidth +
											inputCol

										gradInput.Data[inputIndex] += gradValue * ct.Weights.Data[weightIndex]

										gradWeightsData[weightIndex] += gradValue * inputTensor.Data[inputIndex]
									}
								}
							}
						}
					}
				}
			}
		}
	}

	for i := range ct.Weights.Data {
		ct.Weights.Data[i] -= learningRate * gradWeightsData[i]
	}
	for i := range ct.Bias.Data {
		ct.Bias.Data[i] -= learningRate * gradBiasData[i]
	}

	return gradInput
}

func product(shape []int) int {
	p := 1
	for _, v := range shape {
		p *= v
	}
	return p
}

func (ct *ConvTranspose2dLayer) ZeroGrad() {
}

func (ct *ConvTranspose2dLayer) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{ct.Weights, ct.Bias}
}

func (ct *ConvTranspose2dLayer) SetWeights(data [][]float32) {
	flattenedData := make([]float32, 0)
	for _, row := range data {
		flattenedData = append(flattenedData, row...)
	}

	ct.Weights = tensor.NewTensor(flattenedData, ct.Weights.GetShape())
}

func (ct *ConvTranspose2dLayer) SetBias(data [][]float32) {
	flattenedData := make([]float32, 0)
	for _, row := range data {
		flattenedData = append(flattenedData, row...)
	}
	ct.Bias = tensor.NewTensor(flattenedData, ct.Bias.GetShape())
}

func (l *ConvTranspose2dLayer) SetWeightsAndShape(data []float32, shape []int) {
	l.Weights = tensor.NewTensor(data, shape)
}

func (l *ConvTranspose2dLayer) SetBiasAndShape(data []float32, shape []int) {
	l.Bias = tensor.NewTensor(data, shape)
}
