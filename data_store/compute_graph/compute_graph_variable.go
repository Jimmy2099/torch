package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"sync"
)

var VariableNum = 0

type Variable struct {
	Node
	graph *ComputationalGraph
	value *tensor.Tensor

	//--

	Name string

	Output   *tensor.Tensor
	Grad     *tensor.Tensor
	Children []Node
	mu       sync.Mutex

	OpType   NodeType
	inputs   []*Tensor
	output   *Tensor
	gradFunc func()
}

func newVariable(initValue *tensor.Tensor) *Variable {
	node := &Variable{
		value: initValue,
	}
	VariableNum += 1
	node.Node = &Variable{Name: getName(node) + fmt.Sprint(VariableNum)}
	return node
}

func (v *Variable) Forward() *tensor.Tensor {
	v.Output = v.value
	return v.Output
}

func (v *Variable) Backward(grad *tensor.Tensor) {
	v.mu.Lock()
	defer v.mu.Unlock()
	//v.Grad += grad
	tmp := v.Grad.Add(grad)
	v.Grad.Data = tmp.Data
}

func (v *Variable) SetValue(value *tensor.Tensor) {
	v.value = value
}
