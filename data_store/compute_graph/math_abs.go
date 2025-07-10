package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Abs struct {
	OPS
}

func (m *Abs) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].node.Forward()
	data := make([]float32, len(a.Data))
	for i, v := range a.Data {
		data[i] = float32(math.Abs(float64(v)))
	}

	result := tensor.NewTensor(data, a.GetShape())
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Abs) Backward(grad *tensor.Tensor) {
	aVal := m.Children[0].value

	if aVal == nil || grad == nil {
		panic("nil tensor in abs backward pass")
	}

	signData := make([]float32, len(aVal.Data))
	for i, v := range aVal.Data {
		if v > 0 {
			signData[i] = 1.0
		} else if v < 0 {
			signData[i] = -1.0
		} else {
			signData[i] = 0.0
		}
	}
	signTensor := tensor.NewTensor(signData, aVal.GetShape())
	gradInput := grad.Mul(signTensor)

	m.Children[0].node.Backward(gradInput)
}

func (t *GraphTensor) Abs(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("abs_%d", t.graph.nodeCount)
		t.graph.nodeCount++
	}

	g := t.graph

	node := NewAbs(name, t)

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

func NewAbs(name string, a *GraphTensor) *Abs {
	return &Abs{
		OPS{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}
