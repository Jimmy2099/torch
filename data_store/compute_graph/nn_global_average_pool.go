package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type GlobalAveragePool struct {
	*OPSNode
	OPSTensor
}

func (m *GlobalAveragePool) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	shape := input.GetShape()
	if len(shape) != 4 {
		panic("GlobalAveragePool requires 4D input")
	}

	B, C, H, W := shape[0], shape[1], shape[2], shape[3]
	outputData := make([]float32, B*C)

	for b := 0; b < B; b++ {
		for c := 0; c < C; c++ {
			sum := float32(0.0)
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					idx := b*(C*H*W) + c*(H*W) + h*W + w
					sum += input.Data[idx]
				}
			}
			outputData[b*C+c] = sum / float32(H*W)
		}
	}

	m.output.value = tensor.NewTensor(outputData, []int{B, C})
	m.output.computed = true
	return m.output.value
}

func (m *GlobalAveragePool) Backward(grad *tensor.Tensor) {
	if grad == nil {
		panic("nil gradient in GlobalAveragePool backward pass")
	}

	input := m.Children[0].value
	shape := input.GetShape()
	B, C, H, W := shape[0], shape[1], shape[2], shape[3]
	total := float32(H * W)

	gradInput := make([]float32, len(input.Data))

	for b := 0; b < B; b++ {
		for c := 0; c < C; c++ {
			gradVal := grad.Data[b*C+c] / total
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					idx := b*(C*H*W) + c*(H*W) + h*W + w
					gradInput[idx] = gradVal
				}
			}
		}
	}

	m.Children[0].Node.Backward(tensor.NewTensor(gradInput, input.GetShape()))
}

func (t *GraphTensor) GlobalAveragePool(name string) *GraphTensor {
	if name == "" {
		name = fmt.Sprintf("globalavgpool_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}
	if t.Graph == nil {
		panic("tensor not in graph")
	}

	node := NewGlobalAveragePool(name, t)
	outputShape := []int{t.Shape()[0], t.Shape()[1]}
	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: t.Graph,
		Node:  node,
	}
	outputTensor.SetShape(outputShape)

	t.Graph.Tensors[name] = outputTensor
	node.output = outputTensor
	t.Graph.Nodes = append(t.Graph.Nodes, node)
	return outputTensor
}

func NewGlobalAveragePool(name string, input *GraphTensor) *GlobalAveragePool {
	return &GlobalAveragePool{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "GlobalAveragePool",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{input},
		},
	}
}

func (m *GlobalAveragePool) GetOutput() *GraphTensor {
	return m.output
}
