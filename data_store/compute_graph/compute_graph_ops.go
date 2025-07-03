package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"reflect"
)

//type Constant struct {
//	Variable
//	value *tensor.Tensor
//}

//func NewConstant(name string, value *tensor.Tensor) *Constant {
//	node := &Constant{
//		value: value,
//	}
//	node.BaseNode = Variable{Name: getName(node)}
//	return node
//}

//func (c *Constant) Forward() *tensor.Tensor {
//	c.Output = c.value
//	return c.Output
//}
//
//func (c *Constant) Backward(grad float64) {
//}

//
//type Add struct {
//	Node
//}
//
//func NewAdd(name string, a, b Node) *Add {
//	node := &Add{}
//	node.Node = BaseNode{Name: getName(node)}
//	node.AddChild(a)
//	node.AddChild(b)
//	return node
//}
//
//func (a *Add) Forward() *tensor.Tensor {
//	a.Output = a.Children[0].Forward().Add(a.Children[1].Forward())
//	return a.Output
//}
//
//func (a *Add) Backward(grad *tensor.Tensor) {
//	a.Grad = grad
//	a.Children[0].Backward(grad)
//	a.Children[1].Backward(grad)
//}

type Multiply struct {
	Variable
}

func getName(v interface{}) string {
	multType := reflect.TypeOf(v).Elem()
	return multType.Name()
}

func NewMultiply(name string, a, b Node) *Multiply {
	node := &Multiply{}
	node.Node = &Variable{Name: getName(node)}
	node.AddChild(a)
	node.AddChild(b)
	return node
}

func (m *Multiply) Forward() *tensor.Tensor {
	m.Output = m.Children[0].Forward().Mul(m.Children[1].Forward())
	return m.Output
}

func (m *Multiply) Backward(grad *tensor.Tensor) {
	m.Grad = grad
	a := m.Children[0].GetOutput()
	b := m.Children[1].GetOutput()
	m.Children[0].Backward(b.Mul(grad))
	m.Children[1].Backward(a.Mul(grad))
}

//
//type Sigmoid struct {
//	Node
//}
//
//func NewSigmoid(name string, input NodeInterface) *Sigmoid {
//	node := &Sigmoid{}
//	node.Node = Node{Name: getName(node)}
//	node.AddChild(input)
//	return node
//}
//
//func (s *Sigmoid) Forward() *tensor.Tensor {
//	x := s.Children[0].Forward()
//	s.Output = x.Sigmoid()
//	return s.Output
//}
//
//func (s *Sigmoid) Backward(grad *tensor.Tensor) {
//	s.Grad = grad
//	output := s.Output
//	//localGrad := output * (1 - output)
//	one := tensor.Ones(output.GetShape())
//	localGrad := output.Mul(one.Sub(output))
//	s.Children[0].Backward(localGrad.Mul(grad))
//}
