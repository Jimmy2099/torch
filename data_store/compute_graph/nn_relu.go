package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ReLU struct {
	*OPSNode
	OPSTensor
}

func (m *ReLU) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	result := a.ReLU()
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *ReLU) Backward(grad *tensor.Tensor) {
	x := m.Children[0].value
	gradData := make([]float32, len(x.Data))
	for i, val := range x.Data {
		if val > 0 {
			gradData[i] = grad.Data[i]
		} else {
			gradData[i] = 0
		}
	}
	gradInput := tensor.NewTensor(gradData, x.GetShape())
	m.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) ReLU(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("relu_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewReLU(name, t)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(t.Shape())

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewReLU(name string, a *GraphTensor) *ReLU {
	return &ReLU{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Relu",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *ReLU) GetOutput() *GraphTensor {
	return m.output
}
