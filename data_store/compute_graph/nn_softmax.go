package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Softmax struct {
	*OPSNode
	OPSTensor
}

func (m *Softmax) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	input := a.Copy()

	shape := input.GetShape()
	dims := len(shape)
	if dims == 0 {
		panic("scalar not supported for softmax")
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
			resultData[j] = expVal
			sumExp += expVal
		}

		for j := start; j < end; j++ {
			resultData[j] /= sumExp
		}
	}

	result := tensor.NewTensor(resultData, shape)
	m.output.value = result
	m.output.computed = true
	return result
}

func NewSoftmax(name string, a *GraphTensor) *Softmax {
	return &Softmax{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Softmax",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Softmax) GetOutput() *tensor.Tensor {
	return m.output.value
}
