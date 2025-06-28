package compute_graph

import "github.com/Jimmy2099/torch/data_store/tensor"

type Variable struct {
	BaseNode
	value *tensor.Tensor
}

func NewVariable(name string, initValue *tensor.Tensor) *Variable {
	return &Variable{
		BaseNode: BaseNode{Name: name},
		value:    initValue,
	}
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
