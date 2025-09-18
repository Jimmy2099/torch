package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Constant struct {
	*OPSNode
	OPSTensor
	value *tensor.Tensor
}

func (m *Constant) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	m.output.value = m.value
	m.output.computed = true
	return m.output.value
}

func (m *Constant) Backward(grad *tensor.Tensor) {
}

func (g *ComputationalGraph) Constant(value []float32, shape []int, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("constant_%d", g.NodeCount)
		g.NodeCount++
	}

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor(value, shape),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Shape: shape,
		Graph: g,
	}

	node := NewConstant(name, outputTensor)
	outputTensor.Node = node

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewConstant(name string, output *GraphTensor) *Constant {
	return &Constant{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Constant",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{},
		},
		value: output.value,
	}
}

func (m *Constant) GetOutput() *GraphTensor {
	return m.output
}
