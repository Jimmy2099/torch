package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Greater struct {
	*OPSNode
	OPSTensor
}

func (m *Greater) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	b := m.Children[1].Node.Forward()

	resultData := make([]float32, len(a.Data))
	for i := range a.Data {
		if a.Data[i] > b.Data[i] {
			resultData[i] = 1.0
		} else {
			resultData[i] = 0.0
		}
	}

	result := tensor.NewTensor(resultData, a.GetShape())
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Greater) Backward(grad *tensor.Tensor) {
	return
}

func (t *GraphTensor) Greater(other *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("greater_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewGreater(name, t, other)

	outputShape := t.GetShape()
	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(outputShape)

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewGreater(name string, a, b *GraphTensor) *Greater {
	return &Greater{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Greater",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a, b},
		},
	}
}

func (m *Greater) GetOutput() *tensor.Tensor {
	return m.output.value
}
