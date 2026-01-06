package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Gather struct {
	*OPSNode
	OPSTensor
	Axis int
}

func (m *Gather) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	if len(m.Children) < 2 {
		panic(fmt.Sprintf("Gather %s: 缺少输入, Children 长度: %d", m.Name, len(m.Children)))
	}
	data := m.Children[0].Node.Forward()
	indices := m.Children[1].Node.Forward()
	//if indices == nil {
	//	indices = tensor.Zeros([]int{1})
	//}
	if data == nil || indices == nil {
		panic(fmt.Sprintf("Gather %s input nil: data=%v, indices=%v", m.Name, data, indices))
	}

	result := data.Gather(indices, m.Axis)
	m.output.value = result
	m.output.computed = true
	return result
}

func (t *GraphTensor) Gather(indices *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("gather_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	// 默认使用 axis 0
	node := NewGather(name, t, indices, 0)

	outputShape := calculateGatherOutputShape(t.GetShape(), indices.GetShape(), 0)

	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor(make([]float32, outputSize), outputShape),
		grad:  tensor.NewTensor(make([]float32, outputSize), outputShape),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(outputShape)

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

// 修正：增加 axis 参数，匹配 compute_graph_onnx.go 的调用
func NewGather(name string, data *GraphTensor, indices *GraphTensor, axis int) *Gather {
	return &Gather{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Gather",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{data, indices},
		},
		Axis: axis,
	}
}

func (m *Gather) GetOutput() *tensor.Tensor {
	return m.output.value
}

func calculateGatherOutputShape(dataShape, indicesShape []int, axis int) []int {
	if axis < 0 {
		axis += len(dataShape)
	}
	outputShape := make([]int, 0)
	outputShape = append(outputShape, dataShape[:axis]...)
	outputShape = append(outputShape, indicesShape...)
	if axis+1 < len(dataShape) {
		outputShape = append(outputShape, dataShape[axis+1:]...)
	}
	if len(outputShape) == 0 {
		return []int{1}
	}
	return outputShape
}
