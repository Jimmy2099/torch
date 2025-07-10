package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Sqrt struct {
	OPS
}

func (m *Sqrt) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}
	input := m.Children[0].node.Forward()
	data := make([]float32, len(input.Data))
	for i, v := range input.Data {
		if v < 0 {
			panic("sqrt input must be non-negative")
		}
		data[i] = float32(math.Sqrt(float64(v)))
	}
	m.output.value = tensor.NewTensor(data, input.GetShape())
	m.output.computed = true
	return m.output.value
}

func (m *Sqrt) Backward(grad *tensor.Tensor) {
	outputVal := m.output.value
	if outputVal == nil || grad == nil {
		panic("nil tensor in sqrt backward pass")
	}

	gradData := make([]float32, len(outputVal.Data))
	for i, out := range outputVal.Data {
		gradData[i] = grad.Data[i] / (2 * out)
	}
	gradInput := tensor.NewTensor(gradData, outputVal.GetShape())
	m.Children[0].node.Backward(gradInput)
}

func (t *GraphTensor) Sqrt(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("sqrt_%d", t.graph.nodeCount)
		t.graph.nodeCount++
	}

	g := t.graph

	node := NewSqrt(name, t)

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

func NewSqrt(name string, a *GraphTensor) *Sqrt {
	return &Sqrt{
		OPS{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}
