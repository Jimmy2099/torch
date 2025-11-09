package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	onnx_ir "github.com/Jimmy2099/torch/thirdparty/onnx-go/ir"
)

type Constant struct {
	*OPSNode
	OPSTensor
	value *tensor.Tensor
}

func int642int(in []int64) (out []int) {
	for _, v := range in {
		out = append(out, int(v))
	}
	if len(out) == 0 {
		return nil
	}
	return
}

func (m *Constant) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	{
		attr := ONNXAttr.GetONNXAttributeByName(m.GetName())
		if len(attr) > 0 && attr[0].Name == "value" {
			fmt.Println(attr[0].T.GetInt32Data())
		}
		var v []float32
		fmt.Println(onnx_ir.TensorProto_DataType_name[attr[0].GetT().GetDataType()])
		fmt.Println(getFloatDataFromByte(attr[0].GetT().GetRawData()))

		if onnx_ir.TensorProto_DataType(attr[0].GetT().GetDataType()) == onnx_ir.TensorProto_FLOAT {
			v = getFloatDataFromByte(attr[0].GetT().GetRawData())
		}
		if onnx_ir.TensorProto_DataType(attr[0].GetT().GetDataType()) == onnx_ir.TensorProto_INT64 {
			v = getFloatDataFromInt64Byte(attr[0].GetT().GetRawData())
		}

		fmt.Println("Attr:", attr, "v:", v)
		m.value = tensor.NewTensor(v, int642int(attr[0].GetT().GetDims()))
	}

	m.output.value = m.value
	m.output.computed = true
	return m.output.value
}

func (m *Constant) Backward(grad *tensor.Tensor) {
	return
}

func (g *ComputationalGraph) Constant(value []float32, shape []int, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("constant_%d", g.NodeCount)
		g.NodeCount++
	}

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor(value, shape),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
	}
	outputTensor.SetShape(shape)

	node := NewConstant(name, outputTensor)
	outputTensor.Node = node

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewConstant(name string, output *GraphTensor) *Constant {
	return &Constant{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Constant",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{},
		},
		value: output.value,
	}
}

func (m *Constant) GetOutput() *tensor.Tensor {
	return m.output.value
}
