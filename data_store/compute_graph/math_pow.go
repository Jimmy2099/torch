package compute_graph

import (
	"encoding/binary"
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	math "github.com/chewxy/math32"
)

func (t *GraphTensor) Pow(exponent *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("pow_tensor_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != exponent.Graph {
		panic("tensors belong to different graphs")
	}

	node := NewPow(name, t, exponent)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: t.Graph,
		Node:  node,
	}
	outputTensor.SetShape(t.GetShape())

	t.Graph.Tensors[name] = outputTensor
	node.output = outputTensor
	t.Graph.Nodes = append(t.Graph.Nodes, node)
	return outputTensor
}

type Pow struct {
	*OPSNode
	OPSTensor
}

func NewPow(name string, base, exponent *GraphTensor) *Pow {
	return &Pow{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Pow",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{base, exponent},
		},
	}
}

func getFloatDataFromByte(inData []byte) (outData []float32) {
	{
		count := len(inData) / 4
		outData = make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint32(inData[i*4 : (i+1)*4])
			outData[i] = math.Float32frombits(bits)
		}
	}
	return
}

func getFloatDataFromInt64Byte(inData []byte) (outData []float32) {
	count := len(inData) / 8
	outData = make([]float32, count)
	for i := 0; i < count; i++ {
		bits := binary.LittleEndian.Uint64(inData[i*8 : (i+1)*8])
		f := math.Float64frombits(bits)
		outData[i] = float32(f)
	}
	return
}

func (m *Pow) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	base := m.Children[0].Node.Forward()
	exponent := m.Children[1].Node.Forward()
	fmt.Println(m.GetName())
	fmt.Println("m.Children[0].Node:", m.Children[0].Node.GetName())
	fmt.Println("m.Children[1].Node:", m.Children[1].Node.GetName())

	if len(exponent.Data) != 1 {
		panic("exponent.Data must be of length 1")
	}

	//TODO broadcast
	m.output.value = base.Pow(exponent.Data[0])
	m.output.computed = true
	return m.output.value
}

func (m *Pow) GetOutput() *tensor.Tensor {
	return m.output.value
}
