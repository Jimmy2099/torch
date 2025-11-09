package compute_graph

import onnx_ir "github.com/Jimmy2099/torch/thirdparty/onnx-go/ir"

var ONNXAttr *ONNXAttribute

// ONNXAttribute temporary solution TODO
type ONNXAttribute struct {
	Data map[string][]*onnx_ir.AttributeProto
}

func (m *ONNXAttribute) SetONNXAttribute(name string, data []*onnx_ir.AttributeProto) {
	m.Data[name] = data
}

func (m *ONNXAttribute) GetONNXAttributeByName(name string) []*onnx_ir.AttributeProto {
	return m.Data[name]
}

func NewONNXAttribute() *ONNXAttribute {
	ONNXAttr = &ONNXAttribute{
		Data: map[string][]*onnx_ir.AttributeProto{},
	}
	return ONNXAttr
}
