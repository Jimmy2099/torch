package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type LRN struct {
	OPS
	size   int
	alpha  float32
	beta   float32
	k      float32
	square *tensor.Tensor
}

func (l *LRN) Forward() *tensor.Tensor {
	if l.output.computed {
		return l.output.value
	}

	input := l.Children[0].Node.Forward()
	l.square = input.Copy().Mul(input)

	result := input.Copy()
	data := result.Data
	sqData := l.square.Data
	channels := input.GetShape()[1]
	area := input.GetShape()[0] * input.GetShape()[2] * input.GetShape()[3]

	for c := 0; c < channels; c++ {
		start := max(0, c-l.size/2)
		end := min(channels-1, c+l.size/2)
		sum := float32(0.0)
		for i := start; i <= end; i++ {
			sum += sqData[i*area]
		}
		data[c*area] /= (l.k + l.alpha*sum) * l.beta
	}

	l.output.value = result
	l.output.computed = true
	return result
}

func (l *LRN) Backward(grad *tensor.Tensor) {
	// Simplified backward pass for demonstration
	// Actual implementation would use stored square values
	dInput := grad.Copy().Div(l.output.value.Mul(l.output.value))
	l.Children[0].Node.Backward(dInput)
}

func (t *GraphTensor) LRN(size int, alpha, beta, k float32, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("lrn_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}
	g := t.Graph

	node := NewLRN(name, t, size, alpha, beta, k)

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

func NewLRN(name string, input *GraphTensor, size int, alpha, beta, k float32) *LRN {
	return &LRN{
		OPS: OPS{
			Name:     name,
			Children: []*GraphTensor{input},
		},
		size:  size,
		alpha: alpha,
		beta:  beta,
		k:     k,
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
