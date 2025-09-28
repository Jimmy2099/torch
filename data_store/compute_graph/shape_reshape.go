package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Reshape struct {
	*OPSNode
	OPSTensor
	originalShape []int
	newShape      []int
}

func (m *Reshape) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	m.originalShape = input.GetShape()
	result := input.Reshape(m.newShape)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Reshape) Backward(grad *tensor.Tensor) {
	reshapedGrad := grad.Reshape(m.originalShape)
	m.Children[0].Node.Backward(reshapedGrad)
}

func (t *GraphTensor) Reshape(shape []int, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("reshape_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewReshape(name, t, shape)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(shape)

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewReshape(name string, a *GraphTensor, shape []int) *Reshape {
	return &Reshape{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Reshape",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
		newShape: shape,
	}
}

func (m *Reshape) GetOutput() *GraphTensor {
	return m.output
}
