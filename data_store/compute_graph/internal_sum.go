package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/node"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Sum struct {
	Name     string
	Children []*GraphTensor
	output   *GraphTensor
}

func (m *Sum) GetONNXNodeInfo() *node.ONNXNodeInfo {
	return &node.ONNXNodeInfo{
		Name:           "Sum",
		ProducedTensor: true,
	}
}

func NewSum(name string, a *GraphTensor) *Sum {
	return &Sum{
		Name:     name,
		Children: []*GraphTensor{a},
	}
}

func (m *Sum) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()

	sum := float32(0)
	for _, val := range a.Data {
		sum += val
	}

	result := tensor.NewTensor([]float32{sum}, []int{1})
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Sum) Backward(grad *tensor.Tensor) {
	if len(grad.Data) != 1 {
		panic("gradient for sum must be scalar")
	}

	gradValue := grad.Data[0]
	gradData := make([]float32, len(m.Children[0].value.Data))
	for i := range gradData {
		gradData[i] = gradValue
	}

	gradTensor := tensor.NewTensor(gradData, m.Children[0].value.GetShape())
	m.Children[0].Node.Backward(gradTensor)
}

func (m *Sum) ResetComputed() {
	m.output.computed = false
}

func (m *Sum) GetName() string { return m.Name }

func (m *Sum) GetChildren() []node.Node {
	nodes := make([]node.Node, len(m.Children))
	for i, t := range m.Children {
		nodes[i] = t.Node
	}
	return nodes
}

func (t *GraphTensor) Sum(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("sum_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}
	g := t.Graph

	sumNode := NewSum(name, t)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{0}, []int{1}),
		grad:  tensor.NewTensor([]float32{0}, []int{1}),
		Graph: g,
		Node:  sumNode,
	}
	outputTensor.SetShape(outputTensor.value.GetShape())

	g.Tensors[name] = outputTensor
	sumNode.output = outputTensor
	g.Nodes = append(g.Nodes, sumNode)
	return outputTensor
}

func (m *Sum) GetOutput() *tensor.Tensor {
	return m.output.value
}
