package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/network"
	"github.com/Jimmy2099/torch/data_store/tensor"
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

		var r1Go []*tensor.Tensor
		var r2Onnx []*tensor.Tensor
		var err error

		{
			{
				graphNode.Forward()
				r1Go = append(r1Go, g.GetTensorByName(g.Network.GetNodeByName(graphNode.GetName()).GetOutputName()[0]).Value())
			}

			{
				r2Onnx, err = OnnxNodeCompute(g, g.Network.GetNodeByName(graphNode.GetName()))
				if err != nil {
					panic(err)
				}
			}
		}
		_, _, _ = r1Go, r2Onnx, err
		err = OutPutCompare(r1Go, r2Onnx)
		if err != nil {
			log.Println("ResultCompareByNode error:", err)
			log.Println("set result by onnxruntime compute result!!!")
			t := g.GetTensorByName(g.Network.GetNodeByName(graphNode.GetName()).GetOutputName()[0])
			t.value = r2Onnx[0]
		}
	} else {
		graphNode.Forward()
	}
}
