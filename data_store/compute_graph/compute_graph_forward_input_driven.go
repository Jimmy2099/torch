package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/network"
	"log"
)

func (g *ComputationalGraph) ForwardInputDriven() {
	g.Reset()

	if g.ComputeDependencyGraph.GetOutputSortedNodes() == nil {
		g.ComputeDependencyGraph.ComputeSortedNodes()
	}
	for _, output := range g.ComputeDependencyGraph.GetOutputSortedNodes() {
		g.forwardInputDrivenNode(output)
	}
}

func (g *ComputationalGraph) forwardInputDrivenNode(n *network.Node) {
	if n.IsTensor() {
		return
	}
	graphNode := g.GetNodeByName(n.Name)
	if graphNode == nil {
		panic("graphNode is null: " + n.Name)
	}
	if g.IsDebugMode() {
		graphNode.Forward()
		r1Go, r2Onnx, err := ResultCompareByNode(g, g.Network.GetNodeByName(graphNode.GetName()))
		if err == nil {
			return
		}
		_, _, _ = r1Go, r2Onnx, err
		log.Println("ResultCompareByNode error:", err)
		log.Println("set result by onnxruntime compute result!!!")
		t := g.GetTensorByName(g.Network.GetNodeByName(graphNode.GetName()).GetOutputName()[0])
		t.value = r2Onnx[0]
	} else {
		graphNode.Forward()
	}
}
