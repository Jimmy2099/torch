package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Pad struct {
	*OPSNode
	OPSTensor
	Pads [][2]int
	Val  float32
}

func (m *Pad) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	shape := a.GetShape()
	newShape := make([]int, len(shape))
	for i, s := range shape {
		newShape[i] = s + m.Pads[i][0] + m.Pads[i][1]
	}

	paddedData := make([]float32, product(newShape))
	for i := range paddedData {
		paddedData[i] = m.Val
	}

	copyData(a.Data, paddedData, shape, newShape, m.Pads)

	result := tensor.NewTensor(paddedData, newShape)
	m.output.value = result
	m.output.computed = true
	return result
}

func copyData(src, dst []float32, srcShape, dstShape []int, pads [][2]int) {
	pos := make([]int, len(srcShape))
	for {
		srcIdx := 0
		stride := 1
		for i := len(pos) - 1; i >= 0; i-- {
			srcIdx += pos[i] * stride
			stride *= srcShape[i]
		}

		dstIdx := 0
		stride = 1
		for i := len(pos) - 1; i >= 0; i-- {
			dstIdx += (pos[i] + pads[i][0]) * stride
			stride *= dstShape[i]
		}

		dst[dstIdx] = src[srcIdx]

		for i := len(pos) - 1; i >= 0; i-- {
			pos[i]++
			if pos[i] < srcShape[i] {
				break
			}
			pos[i] = 0
			if i == 0 {
				return
			}
		}
	}
}

func product(shape []int) int {
	p := 1
	for _, s := range shape {
		p *= s
	}
	return p
}

func calculateTotalSize(shape []int) int {
	total := 1
	for _, dim := range shape {
		total *= dim
	}
	return total
}

func (m *Pad) Backward(grad *tensor.Tensor) {
	if grad == nil {
		panic("nil gradient in pad backward pass")
	}

	gradTensor := grad
	for dim := 0; dim < len(m.Pads); dim++ {
		currentShape := gradTensor.GetShape()
		if dim >= len(currentShape) {
			panic(fmt.Sprintf("pad backward: dimension %d out of range (tensor rank %d)", dim, len(currentShape)))
		}
		start := m.Pads[dim][0]
		end := currentShape[dim] - m.Pads[dim][1]
		gradTensor = gradTensor.Slice(start, end, dim)
	}

	m.Children[0].Node.Backward(gradTensor)
}

func (t *GraphTensor) Pad(pads [][2]int, val float32, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("pad_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph

	outputShape := make([]int, len(t.Shape))
	for i, s := range t.Shape {
		outputShape[i] = s + pads[i][0] + pads[i][1]
	}

	totalSize := calculateTotalSize(outputShape)

	valueData := make([]float32, totalSize)
	gradData := make([]float32, totalSize)

	node := NewPad(name, t, pads, val)
	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor(valueData, outputShape),
		grad:  tensor.NewTensor(gradData, outputShape),
		Shape: outputShape,
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

func NewPad(name string, a *GraphTensor, pads [][2]int, val float32) *Pad {
	return &Pad{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Pad",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
		Pads: pads,
		Val:  val,
	}
}

func (m *Pad) GetOutput() *GraphTensor {
	return m.output
}
