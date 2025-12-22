package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Gather struct {
	*OPSNode
	OPSTensor
}

func (m *Gather) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	data := m.Children[0].Node.Forward()
	indices := m.Children[1].Node.Forward()
	//default is '0'
	result := data.Gather(indices, 0)
	m.output.value = result
	m.output.computed = true
	return result
}

func (t *GraphTensor) Gather(indices *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("gather_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewGather(name, t, indices)

	outputShape := calculateGatherOutputShape(t.GetShape(), indices.GetShape())

	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor(make([]float32, outputSize), outputShape),
		grad:  tensor.NewTensor(make([]float32, outputSize), outputShape),
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

func NewGather(name string, data *GraphTensor, indices *GraphTensor) *Gather {
	return &Gather{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Gather",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{data, indices},
		},
	}
}

func (m *Gather) GetOutput() *tensor.Tensor {
	return m.output.value
}

func calculateGatherOutputShape(dataShape, indicesShape []int) []int {
	outputShape := make([]int, 0)
	outputShape = append(outputShape, indicesShape...)
	if len(dataShape) > 1 {
		outputShape = append(outputShape, dataShape[1:]...)
	}
	return outputShape
}
