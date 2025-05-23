package optimizer

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Adam struct {
	Params  []*tensor.Tensor
	LR      float32
	Beta1   float32
	Beta2   float32
	Epsilon float32
	M       []*tensor.Tensor
	V       []*tensor.Tensor
	T       int
}

func NewAdam(params []*tensor.Tensor, lr float32, beta1 float32, beta2 float32, epsilon float32) *Adam {
	m := make([]*tensor.Tensor, len(params))
	v := make([]*tensor.Tensor, len(params))

	for i, p := range params {
		m[i] = tensor.ZerosLike(p)
		v[i] = tensor.ZerosLike(p)
	}

	return &Adam{
		Params:  params,
		LR:      lr,
		Beta1:   beta1,
		Beta2:   beta2,
		Epsilon: epsilon,
		M:       m,
		V:       v,
		T:       0,
	}
}

func (opt *Adam) Step() {
	opt.T++

	beta1t := float32(math.Pow(float64(opt.Beta1), float64(opt.T)))
	beta2t := float32(math.Pow(float64(opt.Beta2), float64(opt.T)))

	for i, p := range opt.Params {
		m := opt.M[i]
		v := opt.V[i]

		for j := range p.Data {
			m.Data[j] = opt.Beta1*m.Data[j] + (1-opt.Beta1)*p.Grad[j]
			v.Data[j] = opt.Beta2*v.Data[j] + (1-opt.Beta2)*p.Grad[j]*p.Grad[j]

			mHat := m.Data[j] / (1 - beta1t)
			vHat := v.Data[j] / (1 - beta2t)

			p.Data[j] -= opt.LR * mHat / (float32(math.Sqrt(float64(vHat))) + opt.Epsilon)
		}
	}
}
