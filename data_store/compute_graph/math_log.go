package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Log struct {
	OPS
}

func (m *Log) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}
	input := m.Children[0].node.Forward()
	data := make([]float32, len(input.Data))
	for i, v := range input.Data {
		if v <= 0 {
			panic("log input must be positive")
		}
		data[i] = float32(math.Log(float64(v)))
	}
	m.output.value = tensor.NewTensor(data, input.GetShape())
	m.output.computed = true
	return m.output.value
}

func (m *Log) Backward(grad *tensor.Tensor) {
	inputVal := m.Children[0].value
	if inputVal == nil || grad == nil {
		panic("nil tensor in log backward pass")
	}

	gradData := make([]float32, len(inputVal.Data))
	for i, v := range inputVal.Data {
		gradData[i] = grad.Data[i] / v
	}
	gradInput := tensor.NewTensor(gradData, inputVal.GetShape())
	m.Children[0].node.Backward(gradInput)
}

func (t *GraphTensor) Log(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("log_%d", t.graph.nodeCount)
		t.graph.nodeCount++
	}

	g := t.graph

	node := NewLog(name, t)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		shape: t.shape,
		graph: g,
		node:  node,
	}

	if _, exists := g.tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.tensors[name] = outputTensor
	node.output = outputTensor
	g.nodes = append(g.nodes, node)
	return outputTensor
}

func NewLog(name string, a *GraphTensor) *Log {
	return &Log{
		OPS{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}
