package optimizer

import "github.com/Jimmy2099/torch/data_store/tensor"

type SGD struct {
	Params []*tensor.Tensor
	LR     float32
}

func NewSGD(params []*tensor.Tensor, lr float32) *SGD {
	return &SGD{
		Params: params,
		LR:     lr,
	}
}

func (opt *SGD) Step() {
	for _, p := range opt.Params {
		for i := range p.Data {
			p.Data[i] -= opt.LR * p.Grad[i]
		}
	}
}
