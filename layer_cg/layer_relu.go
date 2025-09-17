package layer_cg

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/compute_graph"
)

type ReLULayer struct {
	num int
}

func NewReLULayer(graph *compute_graph.ComputationalGraph) *ReLULayer {
	return &ReLULayer{
		num: graph.GetNameAutoInc("NewReLULayer"),
	}
}

func (l *ReLULayer) Forward(x *compute_graph.GraphTensor) *compute_graph.GraphTensor {
	return x.ReLU("layer_relu_" + fmt.Sprint(l.num))
}
