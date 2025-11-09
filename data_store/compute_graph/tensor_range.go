package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/node"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Range struct {
	*OPSNode
	OPSTensor
}

func (m *Range) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	start := m.Children[0].Node.Forward().Data[0]
	limit := m.Children[1].Node.Forward().Data[0]
	delta := m.Children[2].Node.Forward().Data[0]

	var values []float32
	for i := start; i < limit; i += delta {
		values = append(values, i)
	}

	result := tensor.NewTensor(values, []int{len(values)})
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Range) Backward(grad *tensor.Tensor) {
	return
}

func (g *ComputationalGraph) Range(name string, start, limit, delta *GraphTensor) *GraphTensor {
	if len(name) == 0 {
		name = fmt.Sprintf("range_%d", g.NodeCount)
		g.NodeCount++
	}

	node := NewRange(name, start, limit, delta)

	startVal := start.value.Data[0]
	limitVal := limit.value.Data[0]
	deltaVal := delta.value.Data[0]

	count := 0
	for i := startVal; i < limitVal; i += deltaVal {
		count++
	}

	outputShape := []int{count}
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

func NewRange(name string, start, limit, delta *GraphTensor) *Range {
	return &Range{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Range",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{start, limit, delta},
		},
	}
}

func (m *Range) GetOutput() *tensor.Tensor {
	return m.output.value
}

func RangeNodeRegistryFunc(name string, children []*GraphTensor, output *GraphTensor) node.Node {
	if len(children) != 3 {
		panic("Range operation requires exactly 3 inputs")
	}
	m := NewRange(name, children[0], children[1], children[2])
	m.output = output
	return m
}
