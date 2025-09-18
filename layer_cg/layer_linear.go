package layer_cg

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/compute_graph"
)

type LinearLayer struct {
	num         int
	inFeatures  int
	outFeatures int
	weight      *compute_graph.GraphTensor
	bias        *compute_graph.GraphTensor
}

func NewLinearLayer(graph *compute_graph.ComputationalGraph, inFeatures, outFeatures int) *LinearLayer {

	num := graph.GetNameAutoInc("NewLinearLayer")

	return &LinearLayer{
		num:         num,
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
	}
}

func (l *LinearLayer) Forward(x *compute_graph.GraphTensor) *compute_graph.GraphTensor {

	matmul := x.MatMul(l.weight, "linear_matmul_"+fmt.Sprint(l.num))
	output := matmul.Add(l.bias, "linear_add_bias_"+fmt.Sprint(l.num))
	return output
}

func (l *LinearLayer) SetWeight(weight *compute_graph.GraphTensor) {
	l.weight = weight
}

func (l *LinearLayer) SetBias(bias *compute_graph.GraphTensor) {
	l.bias = bias
}
