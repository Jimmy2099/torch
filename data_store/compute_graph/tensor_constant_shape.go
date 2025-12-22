package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ConstantOfShape struct {
	*OPSNode
	OPSTensor
	value float32
}

func (m *ConstantOfShape) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	shapeTensor := m.Children[0].Node.Forward()

	shape := make([]int, len(shapeTensor.Data))
	for i, val := range shapeTensor.Data {
		shape[i] = int(val)
	}

	numElements := 1
	for _, dim := range shape {
		numElements *= dim
	}

	data := make([]float32, numElements)
	for i := range data {
		data[i] = m.value
	}

	result := tensor.NewTensor(data, shape)
	m.output.value = result
	m.output.computed = true
	return result
}

func (t *GraphTensor) ConstantOfShape(names ...string) *GraphTensor {
	return t.ConstantOfShapeWithValue(0.0, names...)
}

func (t *GraphTensor) ConstantOfShapeWithValue(value float32, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("constant_of_shape_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewConstantOfShape(name, t, value)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape([]int{0})

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewConstantOfShape(name string, shapeTensor *GraphTensor, value float32) *ConstantOfShape {
	return &ConstantOfShape{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "ConstantOfShape",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{shapeTensor},
		},
		value: value,
	}
}

func (m *ConstantOfShape) GetOutput() *tensor.Tensor {
	return m.output.value
}
