package tensor

import "github.com/Jimmy2099/torch/pkg/log"

func (t *Tensor) Sub_bak(other *Tensor) *Tensor {
	defer func() {
		if err := recover(); err != nil {
			log.Println("t.shape:", t.shape, " other shape:", other.shape)
			panic(err)
		}
	}()
	result := make([]float32, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] - other.Data[i]
	}
	return &Tensor{Data: result, shape: t.shape}
}

func (t *Tensor) Div_bak(other *Tensor) *Tensor {
	result := make([]float32, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] / other.Data[i]
	}
	return &Tensor{Data: result, shape: t.shape}
}

func (t *Tensor) Mul_bak(other *Tensor) *Tensor {
	result := make([]float32, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] * other.Data[i]
	}
	return &Tensor{Data: result, shape: t.shape}
}

func (t *Tensor) Add_bak(other *Tensor) *Tensor {
	result := make([]float32, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] + other.Data[i]
	}
	return &Tensor{Data: result, shape: t.shape}
}
