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
	OutChannels      int
	inputCache       *tensor.Tensor // Cache for input tensor needed in backward pass
}

func (ct *ConvTranspose2dLayer) GetWeights() *tensor.Tensor {
	return ct.Weights
}

func (ct *ConvTranspose2dLayer) GetBias() *tensor.Tensor {
	return ct.Bias
}

// NewConvTranspose2dLayer creates a new ConvTranspose2dLayer.
func NewConvTranspose2dLayer(inChannels, outChannels, kernelSizeRows, kernelSizeCols, strideRows, strideCols, paddingRows, paddingCols, outputPaddingRows, outputPaddingCols int) *ConvTranspose2dLayer {
	// 修正权重形状为 [in_channels, out_channels, kernel_rows, kernel_cols]
	weightsData := make([]float64, inChannels*outChannels*kernelSizeRows*kernelSizeCols)

	// 修复2: 使用PyTorch一致的Kaiming初始化
	fanIn := inChannels * kernelSizeRows * kernelSizeCols
	variance := 2.0 / float64(fanIn)
	for i := range weightsData {
		weightsData[i] = randn() * math.Sqrt(variance)
	}

	weights := tensor.NewTensor(weightsData, []int{inChannels, outChannels, kernelSizeRows, kernelSizeCols})
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
		OutChannels:      outChannels,
		inputCache:       nil, // Initialize cache to nil
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

func (ct *ConvTranspose2dLayer) Forward(inputTensor *tensor.Tensor) *tensor.Tensor {
	// 输入维度校验
	if len(inputTensor.Shape) != 4 {
		panic(fmt.Sprintf("输入必须是4D张量 [batch, in_channels, height, width], 实际维度: %v", inputTensor.Shape))
	}
	batchSize := inputTensor.Shape[0]
	inChannels := inputTensor.Shape[1]
	inputHeight := inputTensor.Shape[2]
	inputWidth := inputTensor.Shape[3]

	// 参数校验
	if inChannels != ct.InChannels {
		panic(fmt.Sprintf("输入通道数不匹配: 预期 %d 实际 %d", ct.InChannels, inChannels))
	}

	// 计算输出尺寸（与PyTorch公式严格对齐）
	outputHeight := (inputHeight-1)*ct.StrideRow - 2*ct.PaddingRow + ct.KernelSizeRow + ct.OutputPaddingRow
	outputWidth := (inputWidth-1)*ct.StrideCol - 2*ct.PaddingCol + ct.KernelSizeCol + ct.OutputPaddingCol
	if outputHeight <= 0 || outputWidth <= 0 {
		panic(fmt.Sprintf("非法输出尺寸: %dx%d (参数: stride=%dx%d padding=%dx%d kernel=%dx%d output_padding=%dx%d)",
			outputHeight, outputWidth, ct.StrideRow, ct.StrideCol, ct.PaddingRow, ct.PaddingCol,
			ct.KernelSizeRow, ct.KernelSizeCol, ct.OutputPaddingRow, ct.OutputPaddingCol))
	}

	// 初始化输出张量
	outputShape := []int{batchSize, ct.OutChannels, outputHeight, outputWidth}
	outputData := make([]float64, product(outputShape))
	output := tensor.NewTensor(outputData, outputShape)

	// 核心计算逻辑
	for b := 0; b < batchSize; b++ {
		for outC := 0; outC < ct.OutChannels; outC++ {
			for i := 0; i < outputHeight; i++ {
				for j := 0; j < outputWidth; j++ {
					sum := 0.0

					// 遍历卷积核
					for kRow := 0; kRow < ct.KernelSizeRow; kRow++ {
						for kCol := 0; kCol < ct.KernelSizeCol; kCol++ {
							// 计算输入坐标（关键修复点）
							numeratorRow := i + ct.PaddingRow - kRow
							numeratorCol := j + ct.PaddingCol - kCol

							// 检查是否满足步长条件
							if numeratorRow >= 0 && numeratorRow%ct.StrideRow == 0 &&
								numeratorCol >= 0 && numeratorCol%ct.StrideCol == 0 {

								inputRow := numeratorRow / ct.StrideRow
								inputCol := numeratorCol / ct.StrideCol

								// 边界检查
								if inputRow < inputHeight && inputCol < inputWidth {
									// 累加各输入通道的贡献
									for inC := 0; inC < inChannels; inC++ {
										// 权重索引修正（关键修复）
										weightIndex := inC*ct.OutChannels*ct.KernelSizeRow*ct.KernelSizeCol +
											outC*ct.KernelSizeRow*ct.KernelSizeCol +
											kRow*ct.KernelSizeCol +
											kCol

										// 输入索引计算
										inputIndex := b*inChannels*inputHeight*inputWidth +
											inC*inputHeight*inputWidth +
											inputRow*inputWidth +
											inputCol

										// 防御性检查
										if inputIndex < 0 || inputIndex >= len(inputTensor.Data) {
											panic(fmt.Sprintf("输入索引越界: %d (范围 0-%d)", inputIndex, len(inputTensor.Data)))
										}
										if weightIndex < 0 || weightIndex >= len(ct.Weights.Data) {
											panic(fmt.Sprintf("权重索引越界: %d (范围 0-%d)", weightIndex, len(ct.Weights.Data)))
										}

										sum += inputTensor.Data[inputIndex] * ct.Weights.Data[weightIndex]
									}
								}
							}
						}
					}

					// 添加偏置项
					outputIndex := b*ct.OutChannels*outputHeight*outputWidth +
						outC*outputHeight*outputWidth +
						i*outputWidth +
						j

					if outC >= len(ct.Bias.Data) {
						panic(fmt.Sprintf("偏置索引越界: %d (总数 %d)", outC, len(ct.Bias.Data)))
					}
					output.Data[outputIndex] = sum + ct.Bias.Data[outC]
				}
			}
		}
	}

	ct.inputCache = inputTensor
	return output
}

func (ct *ConvTranspose2dLayer) Backward(gradOutput *tensor.Tensor, learningRate float64) *tensor.Tensor {
	if ct.inputCache == nil {
		panic("需要先执行前向传播")
	}
	inputTensor := ct.inputCache

	// 输入参数解析
	batchSize := gradOutput.Shape[0]
	outChannels := gradOutput.Shape[1]
	outputHeight := gradOutput.Shape[2]
	outputWidth := gradOutput.Shape[3]

	inChannels := ct.InChannels
	inputHeight := inputTensor.Shape[2]
	inputWidth := inputTensor.Shape[3]

	// 初始化梯度张量
	gradInputShape := inputTensor.Shape
	gradInputData := make([]float64, product(gradInputShape))
	gradInput := tensor.NewTensor(gradInputData, gradInputShape)

	// 梯度缓冲区
	gradWeightsData := make([]float64, len(ct.Weights.Data))
	gradBiasData := make([]float64, len(ct.Bias.Data))

	// 核心反向计算
	for b := 0; b < batchSize; b++ {
		for outC := 0; outC < outChannels; outC++ {
			for i := 0; i < outputHeight; i++ {
				for j := 0; j < outputWidth; j++ {
					gradOutputIndex := b*outChannels*outputHeight*outputWidth +
						outC*outputHeight*outputWidth +
						i*outputWidth +
						j
					gradValue := gradOutput.Data[gradOutputIndex]

					// 更新偏置梯度
					gradBiasData[outC] += gradValue

					// 遍历卷积核
					for kRow := 0; kRow < ct.KernelSizeRow; kRow++ {
						for kCol := 0; kCol < ct.KernelSizeCol; kCol++ {
							// 计算输入坐标（与前向传播对称）
							numeratorRow := i + ct.PaddingRow - kRow
							numeratorCol := j + ct.PaddingCol - kCol

							if numeratorRow >= 0 && numeratorRow%ct.StrideRow == 0 &&
								numeratorCol >= 0 && numeratorCol%ct.StrideCol == 0 {

								inputRow := numeratorRow / ct.StrideRow
								inputCol := numeratorCol / ct.StrideCol

								if inputRow < inputHeight && inputCol < inputWidth {
									// 计算各输入通道梯度
									for inC := 0; inC < inChannels; inC++ {
										// 权重索引（与前向传播一致）
										weightIndex := inC*ct.OutChannels*ct.KernelSizeRow*ct.KernelSizeCol +
											outC*ct.KernelSizeRow*ct.KernelSizeCol +
											kRow*ct.KernelSizeCol +
											kCol

										// 输入索引
										inputIndex := b*inChannels*inputHeight*inputWidth +
											inC*inputHeight*inputWidth +
											inputRow*inputWidth +
											inputCol

										// 更新输入梯度
										gradInput.Data[inputIndex] += gradValue * ct.Weights.Data[weightIndex]

										// 更新权重梯度（关键修复点）
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

	// 参数更新（可选：建议在优化器中完成）
	for i := range ct.Weights.Data {
		ct.Weights.Data[i] -= learningRate * gradWeightsData[i]
	}
	for i := range ct.Bias.Data {
		ct.Bias.Data[i] -= learningRate * gradBiasData[i]
	}

	return gradInput
}

// 辅助函数：计算切片元素乘积
func product(shape []int) int {
	p := 1
	for _, v := range shape {
		p *= v
	}
	return p
}

// ZeroGrad resets the gradients of weights and biases to zero.
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
