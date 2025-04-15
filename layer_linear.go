package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/viterin/vek/vek32"
	"sync"
)

type LinearLayer struct {
	InputDim          int
	OutputDim         int
	Weights           *tensor.Tensor
	Bias              *tensor.Tensor
	Input             *tensor.Tensor
	Output            *tensor.Tensor
	GradInput         *tensor.Tensor
	WeightDecay       float32
	Momentum          float32
	VWeights          *tensor.Tensor
	VBias             *tensor.Tensor
	WeightsTransposed bool
}

func (l *LinearLayer) GetWeights() *tensor.Tensor {
	return l.Weights
}

func (l *LinearLayer) GetBias() *tensor.Tensor {
	return l.Bias
}

func (l *LinearLayer) SetWeights(data []float32) {
	if len(data) != l.OutputDim*l.InputDim {
		panic("Weights data length mismatch")
	}

	copiedData := make([]float32, len(data))
	copy(copiedData, data)

	l.Weights = tensor.NewTensor(copiedData, []int{l.OutputDim, l.InputDim})
}

func (l *LinearLayer) SetBias(data []float32) {
	if len(data) != l.OutputDim {
		panic("bias data length mismatch")
	}

	copiedData := make([]float32, len(data))
	copy(copiedData, data)

	l.Bias = tensor.NewTensor(copiedData, []int{l.OutputDim, 1})
}

func (l *LinearLayer) SetWeightsAndShape(data []float32, shape []int) {
	l.SetWeights(data)
	l.Weights.Reshape(shape)
}

func (l *LinearLayer) SetBiasAndShape(data []float32, shape []int) {
	l.SetBias(data)
	l.Bias.Reshape(shape)
}

func (l *LinearLayer) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{l.Weights, l.Bias}
}

func NewLinearLayer(inputDim, outputDim int) *LinearLayer {
	weightsData := make([]float32, outputDim*inputDim)
	biasData := make([]float32, outputDim)


	return &LinearLayer{
		InputDim:          inputDim,
		OutputDim:         outputDim,
		Weights:           tensor.NewTensor(weightsData, []int{outputDim, inputDim}),
		Bias:              tensor.NewTensor(biasData, []int{outputDim, 1}),
		VWeights:          tensor.NewTensor(make([]float32, outputDim*inputDim), []int{outputDim, inputDim}),
		VBias:             tensor.NewTensor(make([]float32, outputDim), []int{outputDim, 1}),
		WeightDecay:       0.001,
		Momentum:          0.9,
		WeightsTransposed: false,
	}
}

func (l *LinearLayer) updateParameters(dWeights, dBias *tensor.Tensor, learningRate float32) {
	for i := 0; i < l.Weights.Shape[0]; i++ {
		for j := 0; j < l.Weights.Shape[1]; j++ {
			regGrad := l.WeightDecay * l.Weights.Data[i*l.Weights.Shape[1]+j]
			l.VWeights.Data[i*l.VWeights.Shape[1]+j] = l.Momentum*l.VWeights.Data[i*l.VWeights.Shape[1]+j] -
				learningRate*(dWeights.Data[i*dWeights.Shape[1]+j]+regGrad)
			l.Weights.Data[i*l.Weights.Shape[1]+j] += l.VWeights.Data[i*l.VWeights.Shape[1]+j]
		}
	}

	for i := 0; i < l.Bias.Shape[0]; i++ {
		l.VBias.Data[i] = l.Momentum*l.VBias.Data[i] - learningRate*dBias.Data[i]
		l.Bias.Data[i] += l.VBias.Data[i]
	}
}

func (l *LinearLayer) ZeroGrad() {
	l.GradInput = nil
	l.VWeights = tensor.NewTensor(make([]float32, l.OutputDim*l.InputDim), []int{l.OutputDim, l.InputDim})
	l.VBias = tensor.NewTensor(make([]float32, l.OutputDim), []int{l.OutputDim, 1})
}

func (l *LinearLayer) NumParams() int {
	return l.Weights.Shape[0]*l.Weights.Shape[1] + l.Bias.Shape[0]
}

func (l *LinearLayer) Backward(gradOutput *tensor.Tensor, lr float32) *tensor.Tensor {
	if l.Input == nil || l.Input.Data == nil {
		panic("Forward propagation did not correctly save input data")
	}

	batchSize := gradOutput.Shape[0]
	dWeights := make([]float32, l.InputDim*l.OutputDim)
	dBias := make([]float32, l.OutputDim)
	gradInput := make([]float32, batchSize*l.InputDim)

	for b := 0; b < batchSize; b++ {
		for out := 0; out < l.OutputDim; out++ {
			grad := gradOutput.Data[b*l.OutputDim+out]
			if out >= l.OutputDim || b >= batchSize {
				panic("Gradient index out of bounds")
			}
			for in := 0; in < l.InputDim; in++ {
				dWeights[out*l.InputDim+in] += l.Input.Data[b*l.InputDim+in] * grad
			}
			dBias[out] += grad
		}
	}

	for b := 0; b < batchSize; b++ {
		for in := 0; in < l.InputDim; in++ {
			var sum float32
			for out := 0; out < l.OutputDim; out++ {
				sum += gradOutput.Data[b*l.OutputDim+out] * l.Weights.Data[out*l.InputDim+in]
			}
			gradInput[b*l.InputDim+in] = sum
		}
	}

	if l.VWeights == nil || l.VBias == nil {
		l.VWeights = tensor.NewTensor(make([]float32, len(l.Weights.Data)), l.Weights.Shape)
		l.VBias = tensor.NewTensor(make([]float32, len(l.Bias.Data)), l.Bias.Shape)
	}

	for i := range l.Weights.Data {
		l.VWeights.Data[i] = l.Momentum*l.VWeights.Data[i] - lr*(dWeights[i]/float32(batchSize)+l.WeightDecay*l.Weights.Data[i])
		l.Weights.Data[i] += l.VWeights.Data[i]
	}

	for i := range l.Bias.Data {
		l.VBias.Data[i] = l.Momentum*l.VBias.Data[i] - lr*(dBias[i]/float32(batchSize))
		l.Bias.Data[i] += l.VBias.Data[i]
	}

	return tensor.NewTensor(gradInput, []int{batchSize, l.InputDim})
}

func (l *LinearLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.ForwardSIMD(x)
}

func (l *LinearLayer) ForwardSignalThread(x *tensor.Tensor) *tensor.Tensor {
	originalShape := x.ShapeCopy()
	if len(originalShape) == 0 {
		panic("Input tensor shape cannot be empty")
	}
	inputDim := originalShape[len(originalShape)-1]
	if inputDim != l.InputDim {
		panic(fmt.Sprintf("Input dimension mismatch: last dimension is %d, expected %d", inputDim, l.InputDim))
	}

	flattenedBatch := 1
	for _, dim := range originalShape[:len(originalShape)-1] {
		flattenedBatch *= dim
	}
	reshapedX := x.Reshape([]int{flattenedBatch, l.InputDim})

	l.Input = reshapedX.Clone()
	batchSize := reshapedX.Shape[0]
	outputData := make([]float32, batchSize*l.OutputDim)


	for i := 0; i < batchSize*l.OutputDim*l.InputDim; i++ {
		b := i / (l.OutputDim * l.InputDim)
		remainder := i % (l.OutputDim * l.InputDim)
		out := remainder / l.InputDim
		in := remainder % l.InputDim

		inputIndex := b*l.InputDim + in
		weightIndex := out*l.InputDim + in
		outputIndex := b*l.OutputDim + out

		if in == 0 {
			outputData[outputIndex] = l.Bias.Data[out]
		}
		outputData[outputIndex] += l.Input.Data[inputIndex] * l.Weights.Data[weightIndex]
	}

	newShape := make([]int, len(originalShape))
	copy(newShape, originalShape)
	newShape[len(newShape)-1] = l.OutputDim
	output := tensor.NewTensor(outputData, []int{batchSize, l.OutputDim}).Reshape(newShape)
	l.Output = output

	return output
}

func LinearCompute(Bias, Weights, OutputData []float32, inputLength int, inputFloat32 float32, startPos int) {
	{
		outputDim := len(Bias)
		if outputDim == 0 {
			return
		}
		inputDim := len(Weights) / outputDim
		if inputDim == 0 {
			return
		}
		batchSize := inputLength / inputDim

		totalInputElements := batchSize * inputDim
		adjustedPos := startPos
		if adjustedPos < 0 {
			adjustedPos += ((-adjustedPos-1)/totalInputElements + 1) * totalInputElements
		}
		inputIndex := adjustedPos % totalInputElements

		b := inputIndex / inputDim
		in := inputIndex % inputDim

		for out := 0; out < outputDim; out++ {
			outputIndex := b*outputDim + out
			if outputIndex >= len(OutputData) {
				continue
			}

			weightIndex := out*inputDim + in
			if weightIndex >= len(Weights) {
				continue
			}

			if in == 0 {
				OutputData[outputIndex] = Bias[out]
			}
			OutputData[outputIndex] += inputFloat32 * Weights[weightIndex]
		}
	}
}

func (l *LinearLayer) ForwardSignalThreadCompute(x *tensor.Tensor) *tensor.Tensor {
	originalShape := x.ShapeCopy()
	if len(originalShape) == 0 {
		panic("Input tensor shape cannot be empty")
	}
	inputDim := originalShape[len(originalShape)-1]
	if inputDim != l.InputDim {
		panic(fmt.Sprintf("Input dimension mismatch: last dimension is %d, expected %d", inputDim, l.InputDim))
	}

	flattenedBatch := 1
	for _, dim := range originalShape[:len(originalShape)-1] {
		flattenedBatch *= dim
	}
	reshapedX := x.Reshape([]int{flattenedBatch, l.InputDim})

	l.Input = reshapedX.Clone()
	batchSize := reshapedX.Shape[0]
	outputData := make([]float32, batchSize*l.OutputDim)

	for i := 0; i < len(l.Input.Data); i++ {
		LinearCompute(l.Bias.Data, l.Weights.Data, outputData, len(x.Data), l.Input.Data[i], i)
	}

	newShape := make([]int, len(originalShape))
	copy(newShape, originalShape)
	newShape[len(newShape)-1] = l.OutputDim
	output := tensor.NewTensor(outputData, []int{batchSize, l.OutputDim}).Reshape(newShape)
	l.Output = output

	return output
}

func (l *LinearLayer) ForwardMultiThread(x *tensor.Tensor) *tensor.Tensor {
	originalShape := x.ShapeCopy()
	if len(originalShape) == 0 {
		panic("Input tensor shape cannot be empty")
	}
	inputDim := originalShape[len(originalShape)-1]
	if inputDim != l.InputDim {
		panic(fmt.Sprintf("Input dimension mismatch: last dimension is %d, expected %d", inputDim, l.InputDim))
	}

	flattenedBatch := 1
	for _, dim := range originalShape[:len(originalShape)-1] {
		flattenedBatch *= dim
	}
	reshapedX := x.Reshape([]int{flattenedBatch, l.InputDim})

	l.Input = reshapedX.Clone()
	batchSize := reshapedX.Shape[0]
	outputData := make([]float32, batchSize*l.OutputDim)

	workers := 15
	var wg sync.WaitGroup
	chunkSize := (batchSize + workers - 1) / workers

	for w := 0; w < workers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > batchSize {
			end = batchSize
		}
		if start >= end {
			break
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for b := s; b < e; b++ {
				for out := 0; out < l.OutputDim; out++ {
					sum := l.Bias.Data[out]
					ptr := out * l.InputDim
					for in := 0; in < l.InputDim; in++ {
						sum += l.Input.Data[b*l.InputDim+in] * l.Weights.Data[ptr+in]
					}
					outputData[b*l.OutputDim+out] = sum
				}
			}
		}(start, end)
	}
	wg.Wait()

	newShape := make([]int, len(originalShape))
	copy(newShape, originalShape)
	newShape[len(newShape)-1] = l.OutputDim
	return tensor.NewTensor(outputData, []int{batchSize, l.OutputDim}).Reshape(newShape)
}

func (l *LinearLayer) ForwardSIMD(x *tensor.Tensor) *tensor.Tensor {
	originalShape := x.ShapeCopy()
	if len(originalShape) == 0 {
		panic("Input tensor shape cannot be empty")
	}
	inputDim := originalShape[len(originalShape)-1]
	if inputDim != l.InputDim {
		panic(fmt.Sprintf("Input dimension mismatch: last dimension is %d, expected %d", inputDim, l.InputDim))
	}

	flattenedBatch := 1
	for _, dim := range originalShape[:len(originalShape)-1] {
		flattenedBatch *= dim
	}
	reshapedX := x.Reshape([]int{flattenedBatch, l.InputDim})

	l.Input = reshapedX.Clone()
	batchSize := reshapedX.Shape[0]

	if l.WeightsTransposed == false {
		l.Weights = l.Weights.Transpose()
		l.WeightsTransposed = true
	}

	matmulResult := vek32.MatMul(
		l.Input.Data,
		l.Weights.Data,
		l.InputDim,
	)

	broadcastBias := make([]float32, batchSize*l.OutputDim)
	for b := 0; b < batchSize; b++ {
		copy(
			broadcastBias[b*l.OutputDim:(b+1)*l.OutputDim],
			l.Bias.Data,
		)
	}

	outputData := vek32.Add(matmulResult, broadcastBias)

	newShape := make([]int, len(originalShape))
	copy(newShape, originalShape)
	newShape[len(newShape)-1] = l.OutputDim
	output := tensor.NewTensor(outputData, newShape)
	l.Output = output

	return output
}
