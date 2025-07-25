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
func NewOPSNode(in OPSNode) *OPSNode {
	return &in
}
