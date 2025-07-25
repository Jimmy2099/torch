package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type GlobalMaxPool struct {
	*OPSNode
	OPSTensor
	maxPositions [][]int
}

func (m *GlobalMaxPool) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	shape := input.GetShape()
	if len(shape) != 4 {
		panic("GlobalMaxPool requires 4D input")
	}

	B, C, H, W := shape[0], shape[1], shape[2], shape[3]
	outputData := make([]float32, B*C)
	maxPositions := make([][]int, B)

	for b := range maxPositions {
		maxPositions[b] = make([]int, C)
	}

	for b := 0; b < B; b++ {
		for c := 0; c < C; c++ {
			maxVal := float32(-1e9)
			maxPos := -1
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					idx := b*(C*H*W) + c*(H*W) + h*W + w
					val := input.Data[idx]
					if val > maxVal {
						maxVal = val
						maxPos = h*W + w
					}
				}
			}
			outputData[b*C+c] = maxVal
			maxPositions[b][c] = maxPos
		}
	}

	m.output.value = tensor.NewTensor(outputData, []int{B, C})
	m.output.computed = true
	m.maxPositions = maxPositions
	return m.output.value
}

func (m *GlobalMaxPool) Backward(grad *tensor.Tensor) {
	if grad == nil {
		panic("nil gradient in GlobalMaxPool backward pass")
	}

	input := m.Children[0].value
	shape := input.GetShape()
	B, C, H, W := shape[0], shape[1], shape[2], shape[3]

	gradInput := make([]float32, len(input.Data))

	for b := 0; b < B; b++ {
		for c := 0; c < C; c++ {
			pos := m.maxPositions[b][c]
			if pos != -1 {
				gradInput[b*(C*H*W)+c*(H*W)+pos] = grad.Data[b*C+c]
			}
		}
	}

	m.Children[0].Node.Backward(tensor.NewTensor(gradInput, input.GetShape()))
}

func (t *GraphTensor) GlobalMaxPool(name string) *GraphTensor {
	if name == "" {
		name = fmt.Sprintf("globalmaxpool_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}
	if t.Graph == nil {
		panic("tensor not in graph")
	}

	node := NewGlobalMaxPool(name, t)
	outputShape := []int{t.Shape[0], t.Shape[1]}
	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Shape: outputShape,
		Graph: t.Graph,
		Node:  node,
	}

	t.Graph.Tensors[name] = outputTensor
	node.output = outputTensor
	t.Graph.Nodes = append(t.Graph.Nodes, node)
	return outputTensor
}

func NewGlobalMaxPool(name string, input *GraphTensor) *GlobalMaxPool {
	return &GlobalMaxPool{
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{input},
		},
	}
}

func (m *GlobalMaxPool) GetOutput() *GraphTensor {
	return m.output
}
