package compute_graph

import onnx_ir "github.com/Jimmy2099/torch/thirdparty/onnx-go/ir"

var ONNXAttrPool *ONNXAttributePool

// ONNXAttributePool temporary solution TODO
type ONNXAttributePool struct {
	Data map[string][]*onnx_ir.AttributeProto
}

func (m *ONNXAttributePool) SetONNXAttribute(name string, data []*onnx_ir.AttributeProto) {
	m.Data[name] = data
}

func (m *ONNXAttributePool) GetONNXAttributeByName(name string) []*onnx_ir.AttributeProto {
	return m.Data[name]
}

func NewONNXAttributePool() *ONNXAttributePool {
	ONNXAttrPool = &ONNXAttributePool{
		Data: map[string][]*onnx_ir.AttributeProto{},
	}
	return ONNXAttrPool
}
