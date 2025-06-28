package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Constant struct {
	BaseNode
	value *tensor.Tensor
}

func NewConstant(name string, value *tensor.Tensor) *Constant {
	return &Constant{
		BaseNode: BaseNode{Name: name},
		value:    value,
	}
}

func (c *Constant) Forward() *tensor.Tensor {
	c.Output = c.value
	return c.Output
}

func (c *Constant) Backward(grad float64) {
}

type Add struct {
	BaseNode
}

func NewAdd(name string, a, b Node) *Add {
	node := &Add{BaseNode: BaseNode{Name: name}}
	node.AddChild(a)
	node.AddChild(b)
	return node
}

func (a *Add) Forward() *tensor.Tensor {
	a.Output = a.Children[0].Forward().Add(a.Children[1].Forward())
	return a.Output
}

func (a *Add) Backward(grad *tensor.Tensor) {
	a.Grad = grad
	a.Children[0].Backward(grad)
	a.Children[1].Backward(grad)
}

type Multiply struct {
	BaseNode
}

func NewMultiply(name string, a, b Node) *Multiply {
	node := &Multiply{BaseNode: BaseNode{Name: name}}
	node.AddChild(a)
	node.AddChild(b)
	return node
}

func (m *Multiply) Forward() *tensor.Tensor {
	m.Output = m.Children[0].Forward().Multiply(m.Children[1].Forward())
	return m.Output
}

func (m *Multiply) Backward(grad *tensor.Tensor) {
	m.Grad = grad
	a := m.Children[0].GetOutput()
	b := m.Children[1].GetOutput()
	m.Children[0].Backward(b.Multiply(grad))
	m.Children[1].Backward(a.Multiply(grad))
}

type Sigmoid struct {
	BaseNode
}

func NewSigmoid(name string, input Node) *Sigmoid {
	node := &Sigmoid{BaseNode: BaseNode{Name: name}}
	node.AddChild(input)
	return node
}

func (s *Sigmoid) Forward() *tensor.Tensor {
	x := s.Children[0].Forward()
	s.Output = x.Sigmoid()
	return s.Output
}

func (s *Sigmoid) Backward(grad *tensor.Tensor) {
	s.Grad = grad
	output := s.Output
	//localGrad := output * (1 - output)
	one := tensor.Ones(output.GetShape())
	localGrad := output.Multiply(one.Sub(output))
	s.Children[0].Backward(grad.Multiply(localGrad))
}
