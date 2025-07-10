package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Sum struct {
	Name     string
	Children []*GraphTensor
	output   *GraphTensor
}

func NewSum(name string, a *GraphTensor) *Sum {
	return &Sum{
		Name:     name,
		Children: []*GraphTensor{a},
	}
}

func (s *Sum) Forward() *tensor.Tensor {
	if s.output.computed {
		return s.output.value
	}

	a := s.Children[0].node.Forward()

	sum := float32(0)
	for _, val := range a.Data {
		sum += val
	}

	result := tensor.NewTensor([]float32{sum}, []int{1})
	s.output.value = result
	s.output.computed = true
	return result
}

func (s *Sum) Backward(grad *tensor.Tensor) {
	if len(grad.Data) != 1 {
		panic("gradient for sum must be scalar")
	}

	gradValue := grad.Data[0]
	gradData := make([]float32, len(s.Children[0].value.Data))
	for i := range gradData {
		gradData[i] = gradValue
	}

	gradTensor := tensor.NewTensor(gradData, s.Children[0].value.GetShape())
	s.Children[0].node.Backward(gradTensor)
}

func (s *Sum) ResetComputed() {
	s.output.computed = false
}

func (s *Sum) GetName() string { return s.Name }
func (s *Sum) GetChildren() []Node {
	nodes := make([]Node, len(s.Children))
	for i, t := range s.Children {
		nodes[i] = t.node
	}
	return nodes
}

func (t *GraphTensor) Sum(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("sum_%d", t.graph.nodeCount)
		t.graph.nodeCount++
	}
	g := t.graph

	sumNode := NewSum(name, t)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{0}, []int{1}), // Initialize with scalar
		grad:  tensor.NewTensor([]float32{0}, []int{1}),
		shape: []int{1},
		graph: g,
		node:  sumNode,
	}

	g.tensors[name] = outputTensor
	sumNode.output = outputTensor
	g.nodes = append(g.nodes, sumNode)
	return outputTensor
}
