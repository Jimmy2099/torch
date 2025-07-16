package compute_graph

import (
	"fmt"
	"math"

	"github.com/Jimmy2099/torch/data_store/tensor"
)

type LpNormalization struct {
	OPS
	p       float32
	axis    int
	epsilon float32
	norm    *tensor.Tensor
}

func (m *LpNormalization) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	shape := input.GetShape()
	if m.axis < 0 || m.axis >= len(shape) {
		panic("invalid axis for LpNormalization")
	}

	normData := make([]float32, len(input.Data))
	outputData := make([]float32, len(input.Data))

	stride := 1
	for i := m.axis + 1; i < len(shape); i++ {
		stride *= shape[i]
	}
	step := stride * shape[m.axis]
	count := shape[m.axis]

	for offset := 0; offset < len(input.Data); offset += step {
		end := offset + step
		if end > len(input.Data) {
			end = len(input.Data)
		}

		for i := offset; i < end; i += stride {
			sliceEnd := i + stride*count
			if sliceEnd > end {
				sliceEnd = end
			}

			sum := float32(0.0)
			for j := i; j < sliceEnd; j += stride {
				val := float32(math.Pow(float64(math.Abs(float64(input.Data[j]))), float64(m.p)))
				sum += val
			}
			norm := float32(math.Pow(float64(sum), 1/float64(m.p)))
			norm += m.epsilon

			for j := i; j < sliceEnd; j += stride {
				normData[j] = norm
				outputData[j] = input.Data[j] / norm
			}
		}
	}

	m.norm = tensor.NewTensor(normData, shape)
	m.output.value = tensor.NewTensor(outputData, shape)
	m.output.computed = true
	return m.output.value
}

func (m *LpNormalization) Backward(grad *tensor.Tensor) {
	input := m.Children[0].value
	shape := input.GetShape()
	stride := 1
	for i := m.axis + 1; i < len(shape); i++ {
		stride *= shape[i]
	}
	step := stride * shape[m.axis]
	count := shape[m.axis]

	gradInput := make([]float32, len(input.Data))

	for offset := 0; offset < len(input.Data); offset += step {
		end := offset + step
		if end > len(input.Data) {
			end = len(input.Data)
		}

		for i := offset; i < end; i += stride {
			sliceEnd := i + stride*count
			if sliceEnd > end {
				sliceEnd = end
			}

			sum1 := float32(0.0)
			sum2 := float32(0.0)
			for j := i; j < sliceEnd; j += stride {
				sum1 += grad.Data[j]
				sum2 += grad.Data[j] * input.Data[j]
			}

			for j := i; j < sliceEnd; j += stride {
				norm := m.norm.Data[j]
				x := input.Data[j]
				term1 := grad.Data[j] / norm
				term2 := x * float32(math.Pow(math.Abs(float64(x)), float64(m.p-1))) *
					sum2 / (norm * norm * norm)
				gradInput[j] = term1 - term2
			}
		}
	}

	m.Children[0].Node.Backward(tensor.NewTensor(gradInput, shape))
}

func (t *GraphTensor) LpNormalization(p float32, axis int, epsilon float32, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("lpnorm_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewLpNormalization(name, t, p, axis, epsilon)

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

func NewLpNormalization(name string, input *GraphTensor, p float32, axis int, epsilon float32) *LpNormalization {
	return &LpNormalization{
		OPS: OPS{
			Name:     name,
			Children: []*GraphTensor{input},
		},
		p:       p,
		axis:    axis,
		epsilon: epsilon,
	}
}
