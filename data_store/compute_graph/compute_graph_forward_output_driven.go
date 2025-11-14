package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/network"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"log"
	"strings"
)

func (g *ComputationalGraph) ForwardOutputDriven() {
	g.Reset()

	if len(g.Network.GetOutput()) == 0 {
		panic("Error: Computational graph output is not set.")
	}
	visited := make(map[*network.Node]bool)

	for _, output := range g.Network.GetOutput() {
		g.forwardOutputDrivenNode(output, visited)
	}
}

func (g *ComputationalGraph) forwardOutputDrivenNode(n *network.Node, visited map[*network.Node]bool) {
	if n == nil {
		return
	}
	if visited[n] {
		return
	}
	visited[n] = true

	if strings.LastIndex(n.Type, "Tensor_") == 0 || len(n.Outputs) == 0 {
		if n.Inputs != nil { //if n.Parent != nil {
			g.forwardOutputDrivenNode(n.Inputs[0], visited)
		} else {
		}
		return
	}

	for _, in := range n.Inputs {
		if in == nil {
			continue
		}
		g.forwardOutputDrivenNode(in, visited)
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
