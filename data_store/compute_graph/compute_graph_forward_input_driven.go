package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/network"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
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

		var resultGo []*tensor.Tensor
		var resultOnnx []*tensor.Tensor
		var err error
		{
			fmt.Println("--------------------")
			fmt.Println("OPERATOR: ", n.Name)
		}
		{
			{
				resultOnnx, err = OnnxNodeCompute(g, g.Network.GetNodeByName(graphNode.GetName()))
				if err != nil {
					panic(err)
				}
			}
			{
				if false {
					resultGo = append(resultGo, nil)
				} else {
					graphNode.Forward()
					resultGo = append(resultGo, g.GetTensorByName(g.Network.GetNodeByName(graphNode.GetName()).GetOutputName()[0]).Value())
				}
			}
		}
		_, _, _ = resultGo, resultOnnx, err
		err = OutPutCompare(resultGo, resultOnnx)
		if err != nil {
			log.Println("ResultCompareByNode error:", err)
			log.Println("set result by onnxruntime compute result!!!")
			t := g.GetTensorByName(g.Network.GetNodeByName(graphNode.GetName()).GetOutputName()[0])
			t.value = resultOnnx[0]
		}
	} else {
		graphNode.Forward()
	}
}
