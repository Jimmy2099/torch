package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Sigmoid struct {
	*OPSNode
	OPSTensor
}

func (m *Sigmoid) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	result := a.Sigmoid()
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Sigmoid) Backward(grad *tensor.Tensor) {
	y := m.output.value
	gradData := make([]float32, len(y.Data))
	for i, val := range y.Data {
		gradData[i] = grad.Data[i] * val * (1 - val)
	}
	gradInput := tensor.NewTensor(gradData, y.GetShape())
	m.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) Sigmoid(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("sigmoid_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewSigmoid(name, t)

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

func NewSigmoid(name string, a *GraphTensor) *Sigmoid {
	return &Sigmoid{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Sigmoid",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Sigmoid) GetOutput() *GraphTensor {
	return m.output
}
