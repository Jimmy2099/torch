package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Hardmax struct {
	*OPSNode
	OPSTensor
}

func (m *Hardmax) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	input := a.Copy()

	shape := input.GetShape()
	dims := len(shape)
	if dims == 0 {
		panic("scalar not supported for hardmax")
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
		maxIndex := start
		for j := start; j < end; j++ {
			if input.Data[j] > input.Data[maxIndex] {
				maxIndex = j
			}
		}
		for j := start; j < end; j++ {
			if j == maxIndex {
				resultData[j] = 1.0
			} else {
				resultData[j] = 0.0
			}
		}
	}

	result := tensor.NewTensor(resultData, shape)
	m.output.value = result
	m.output.computed = true
	return result
}

func NewHardmax(name string, a *GraphTensor) *Hardmax {
	return &Hardmax{
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Hardmax) GetOutput() *tensor.Tensor {
	return m.output.value
}
