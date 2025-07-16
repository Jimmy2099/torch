package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type LogSoftmax struct {
	OPS
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

func (m *LogSoftmax) Backward(grad *tensor.Tensor) {
	if grad == nil {
		panic("nil gradient in logsoftmax backward")
	}

	s := m.softmax
	shape := s.GetShape()
	dims := len(shape)
	var lastDim int
	if dims == 1 {
		lastDim = shape[0]
	} else if dims == 2 {
		lastDim = shape[1]
	}

	total := len(s.Data)
	numRows := total / lastDim
	sumData := make([]float32, total)

	for i := 0; i < numRows; i++ {
		start := i * lastDim
		end := start + lastDim
		sum := float32(0)
		for j := start; j < end; j++ {
			sum += grad.Data[j]
		}
		for j := start; j < end; j++ {
			sumData[j] = sum
		}
	}

	sumTensor := tensor.NewTensor(sumData, shape)
	sMulSum := s.Copy().Mul(sumTensor)
	gradInput := grad.Copy().Sub(sMulSum)

	m.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) LogSoftmax(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("logsoftmax_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewLogSoftmax(name, t)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Shape: t.Shape,
		Graph: g,
		Node:  node,
	}

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewLogSoftmax(name string, a *GraphTensor) *LogSoftmax {
	return &LogSoftmax{
		OPS: OPS{
			Name:     name,
			Children: []*GraphTensor{a},
		},
		softmax: nil,
	}
}
