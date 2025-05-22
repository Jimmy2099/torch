package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"sync"
)

type LinearLayer struct {
	InputDim          int
	OutputDim         int
	Weights           *tensor.Tensor
	Bias              *tensor.Tensor
	WeightsTransposed bool
}

func (l *LinearLayer) ZeroGrad() {
	l.Weights.ZeroGrad()
	l.Bias.ZeroGrad()
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
	l.Weights.EnableGrad()
}

func (l *LinearLayer) SetBias(data []float32) {
	if len(data) != l.OutputDim {
		panic("bias data length mismatch")
	}

	copiedData := make([]float32, len(data))
	copy(copiedData, data)

	l.Bias = tensor.NewTensor(copiedData, []int{l.OutputDim, 1})
	l.Bias.EnableGrad()
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
		WeightsTransposed: false,
	}
}
func (l *LinearLayer) Backward(x *tensor.Tensor, lr float32) *tensor.Tensor {
	return nil
}

func (l *LinearLayer) SetupBackward(x, out *tensor.Tensor) {

	out.EnableGrad()
	l.Weights.EnableGrad()
	l.Bias.EnableGrad()
	out.Parents = []*tensor.Tensor{x, l.Weights, l.Bias}
	out.GradFn = func() {
		batchSize := x.GetShape()[0]
		inputDim := l.InputDim
		outputDim := l.OutputDim

		if x.RequireGrad() {
			for b := 0; b < batchSize; b++ {
				for j := 0; j < inputDim; j++ {
					dx := float32(0)
					for i := 0; i < outputDim; i++ {
						dx += out.Grad[b*outputDim+i] * l.Weights.Data[i*inputDim+j]
					}
					x.Grad[b*inputDim+j] += dx
				}
			}
		}

		if l.Weights.RequireGrad() {
			for b := 0; b < batchSize; b++ {
				for i := 0; i < outputDim; i++ {
					for j := 0; j < inputDim; j++ {
						l.Weights.Grad[i*inputDim+j] += out.Grad[b*outputDim+i] * x.Data[b*inputDim+j]
					}
				}
			}
		}

		if l.Bias.RequireGrad() {
			for b := 0; b < batchSize; b++ {
				for i := 0; i < outputDim; i++ {
					l.Bias.Grad[i] += out.Grad[b*outputDim+i]
				}
			}
		}
	}
}

func (l *LinearLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	out := l.ForwardSignalThreadCompute(x)
	l.SetupBackward(x, out)
	return out
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

	input := reshapedX.Clone()
	batchSize := reshapedX.GetShape()[0]
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
		outputData[outputIndex] += input.Data[inputIndex] * l.Weights.Data[weightIndex]
	}

	newShape := make([]int, len(originalShape))
	copy(newShape, originalShape)
	newShape[len(newShape)-1] = l.OutputDim
	output := tensor.NewTensor(outputData, []int{batchSize, l.OutputDim}).Reshape(newShape)

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

	input := reshapedX.Clone()
	batchSize := reshapedX.GetShape()[0]
	outputData := make([]float32, batchSize*l.OutputDim)

	for i := 0; i < len(input.Data); i++ {
		LinearCompute(l.Bias.Data, l.Weights.Data, outputData, len(x.Data), input.Data[i], i)
	}

	newShape := make([]int, len(originalShape))
	copy(newShape, originalShape)
	newShape[len(newShape)-1] = l.OutputDim
	output := tensor.NewTensor(outputData, []int{batchSize, l.OutputDim}).Reshape(newShape)

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

	input := reshapedX.Clone()
	batchSize := reshapedX.GetShape()[0]
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
						sum += input.Data[b*l.InputDim+in] * l.Weights.Data[ptr+in]
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
