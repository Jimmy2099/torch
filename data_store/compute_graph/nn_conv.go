package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Conv struct {
	OPS
	StrideH int
	StrideW int
	PadH    int
	PadW    int
}

func (m *Conv) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	weight := m.Children[1].Node.Forward()

	result, err := input.Conv2D(
		weight,
		m.StrideH,
		m.StrideW,
		m.PadH,
		m.PadW,
	)
	if err != nil {
		panic(err)
	}
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Conv) Backward(grad *tensor.Tensor) {
	input := m.Children[0].value
	weight := m.Children[1].value

	if input == nil || weight == nil || grad == nil {
		panic("nil tensor in convolution backward pass")
	}

	gradInput, _ := grad.Conv2D(
		weight,
		m.StrideH,
		m.StrideW,
		m.PadH,
		m.PadW,
	)
	gradWeight, _ := input.Conv2D(
		grad,
		m.StrideH,
		m.StrideW,
		m.PadH,
		m.PadW,
	)

	m.Children[0].Node.Backward(gradInput)
	m.Children[1].Node.Backward(gradWeight)
}

func (t *GraphTensor) Conv(weight *GraphTensor, stride, padding []int, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("conv_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != weight.Graph {
		panic("tensors belong to different graphs")
	}
	g := t.Graph

	sH, sW, padH, padW := processConvParams(stride, padding)

	node := NewConv(name, sH, sW, padH, padW, t, weight)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Shape: []int{0},
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

func processConvParams(stride, padding []int) (sH, sW, padH, padW int) {
	switch len(stride) {
	case 1:
		sH = stride[0]
		sW = stride[0]
	case 2:
		sH = stride[0]
		sW = stride[1]
	default:
		panic("stride must have 1 or 2 elements")
	}

	switch len(padding) {
	case 0:
		padH = 0
		padW = 0
	case 1:
		padH = padding[0]
		padW = padding[0]
	case 2:
		padH = padding[0]
		padW = padding[1]
	default:
		panic("padding must have 0, 1, or 2 elements")
	}
	return
}

func NewConv(name string, strideH, strideW, padH, padW int, input, weight *GraphTensor) *Conv {
	return &Conv{
		OPS: OPS{
			Name:     name,
			Children: []*GraphTensor{input, weight},
		},
		StrideH: strideH,
		StrideW: strideW,
		PadH:    padH,
		PadW:    padW,
	}
}
