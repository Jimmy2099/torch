package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type LogSoftmax struct {
	*OPSNode
	OPSTensor
	softmax *tensor.Tensor
}

func (m *LogSoftmax) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	input := a.Copy()

	shape := input.GetShape()
	dims := len(shape)
	if dims == 0 {
		panic("scalar not supported for logsoftmax")
	}
	var lastDim int
	if dims == 1 {
		lastDim = shape[0]
	} else if dims == 2 {
		lastDim = shape[1]
	} else {
		panic("only 1D and 2D tensors supported")
	}

	total := len(input.Data)
	numRows := total / lastDim
	softmaxData := make([]float32, total)
	resultData := make([]float32, total)

	for i := 0; i < numRows; i++ {
		start := i * lastDim
		end := start + lastDim
		maxVal := input.Data[start]
		for j := start; j < end; j++ {
			if input.Data[j] > maxVal {
				maxVal = input.Data[j]
			}
		}

		sumExp := float32(0)
		for j := start; j < end; j++ {
			expVal := float32(math.Exp(float64(input.Data[j] - float32(maxVal))))
			softmaxData[j] = expVal
			sumExp += expVal
		}

		logSum := float32(math.Log(float64(sumExp)))
		for j := start; j < end; j++ {
			resultData[j] = input.Data[j] - maxVal - logSum
		}
	}

	m.softmax = tensor.NewTensor(softmaxData, shape)
	result := tensor.NewTensor(resultData, shape)
	m.output.value = result
	m.output.computed = true
	return result
}

func NewLogSoftmax(name string, a *GraphTensor) *LogSoftmax {
	return &LogSoftmax{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "LogSoftmax",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
		softmax: nil,
	}
}

func (m *LogSoftmax) GetOutput() *tensor.Tensor {
	return m.output.value
}
