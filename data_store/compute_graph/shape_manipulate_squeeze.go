package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Squeeze struct {
	OPS
	originalShape []int
}

func (s *Squeeze) Forward() *tensor.Tensor {
	if s.output.computed {
		return s.output.value
	}

	input := s.Children[0].Node.Forward()
	s.originalShape = input.GetShape()

	newShape := []int{}
	for _, dim := range s.originalShape {
		if dim != 1 {
			newShape = append(newShape, dim)
		}
	}

	result := input.Reshape(newShape)
	s.output.value = result
	s.output.computed = true
	return result
}

func (s *Squeeze) Backward(grad *tensor.Tensor) {
	if grad == nil {
		panic("nil gradient in squeeze backward pass")
	}
	gradInput := grad.Reshape(s.originalShape)
	s.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) Squeeze(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("squeeze_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewSqueeze(name, t)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Shape: node.Children[0].Shape, // Will be updated during forward
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

func NewSqueeze(name string, a *GraphTensor) *Squeeze {
	return &Squeeze{
		OPS: OPS{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}
