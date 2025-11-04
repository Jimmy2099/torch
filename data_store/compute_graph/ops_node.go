package compute_graph

import "github.com/Jimmy2099/torch/data_store/node"

type OPSNode struct {
	ONNXName           string
	ONNXProducedTensor bool
}

func (m *OPSNode) GetONNXNodeInfo() *node.ONNXNodeInfo {
	return &node.ONNXNodeInfo{
		Name:           m.ONNXName,
		ProducedTensor: m.ONNXProducedTensor,
	}
}

var onnxOPSNameMap map[string]*ONNXOperator

func init() {
	onnxOPSNameMap = map[string]*ONNXOperator{}
	for i := 0; i < len(ONNXOperators); i++ {
		onnxOPSNameMap[ONNXOperators[i].Name] = &ONNXOperators[i]
		for j := 0; j < len(onnxOPSNameMap[ONNXOperators[i].Name].AliasList); j++ {
			onnxOPSNameMap[onnxOPSNameMap[ONNXOperators[i].Name].AliasList[j]] = &ONNXOperators[i]
		}
	}
}

func NewOPSNode(in OPSNode) *OPSNode {
	if _, ok := onnxOPSNameMap[in.ONNXName]; !ok {
		panic("invalid ONNX OPS Name")
	}
	return &in
}

func GetONNXNodeInfoByName(name string) *ONNXOperator {
	return onnxOPSNameMap[name]
}
