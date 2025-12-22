package compute_graph

import (
	"fmt"
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

func (t *GraphTensor) Hardmax(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("hardmax_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewHardmax(name, t)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(t.GetShape())

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
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
