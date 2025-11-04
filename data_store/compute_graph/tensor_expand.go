package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Expand struct {
	*OPSNode
	OPSTensor
}

func (m *Expand) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	shapeTensor := m.Children[1].Node.Forward()

	shape := make([]int, len(shapeTensor.Data))
	for i, val := range shapeTensor.Data {
		shape[i] = int(val)
	}

	inputShape := input.GetShape()
	if len(inputShape) > len(shape) {
		panic(fmt.Sprintf("input has more dimensions (%d) than target shape (%d)", len(inputShape), len(shape)))
	}

	paddedInputShape := make([]int, len(shape))
	for i := range shape {
		if i < len(inputShape) {
			paddedInputShape[i] = inputShape[i]
		} else {
			paddedInputShape[i] = 1
		}
	}

	for i := range shape {
		if paddedInputShape[i] != shape[i] && paddedInputShape[i] != 1 {
			panic(fmt.Sprintf("incompatible dimensions: input has %d, target has %d at dimension %d",
				paddedInputShape[i], shape[i], i))
		}
	}

	result := input.Expand(shape)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Expand) Backward(grad *tensor.Tensor) {
	inputShape := m.Children[0].Node.GetOutput().GetShape()
	outputShape := grad.GetShape()

	var reduceDims []int
	for i := 0; i < len(outputShape); i++ {
		if i >= len(inputShape) {
			reduceDims = append(reduceDims, i)
		} else if inputShape[i] == 1 && outputShape[i] > 1 {
			reduceDims = append(reduceDims, i)
		}
	}

	summedGrad := grad.SumByDim1(reduceDims, true)

	reshapedGrad := summedGrad.Reshape(inputShape)
	m.Children[0].Node.Backward(reshapedGrad)
}

func (t *GraphTensor) Expand(shapeTensor *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("expand_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewExpand(name, t, shapeTensor)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape([]int{0})

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewExpand(name string, input *GraphTensor, shape *GraphTensor) *Expand {
	return &Expand{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Expand",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{input, shape},
		},
	}
}

func (m *Expand) GetOutput() *tensor.Tensor {
	return m.output.value
}
