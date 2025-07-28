package compute_graph

type OPSNode struct {
	ONNXName           string
	ONNXProducedTensor bool
}

func (m *OPSNode) GetONNXNodeInfo() *ONNXNodeInfo {
	return &ONNXNodeInfo{
		Name:           m.ONNXName,
		ProducedTensor: m.ONNXProducedTensor,
	}
}

//	func (m *OPSNode) GetOutput() *GraphTensor {
//		return nil
//	}
var onnxOPSNameMap map[string]interface{}

func init() {
	onnxOPSNameMap = map[string]interface{}{}
	for i := 0; i < len(ONNXOperators); i++ {
		onnxOPSNameMap[ONNXOperators[i].Name] = nil
	}
}

func NewOPSNode(in OPSNode) *OPSNode {
	if _, ok := onnxOPSNameMap[in.ONNXName]; !ok {
		panic("invalid ONNX OPS Name")
	}
	return &in
}
