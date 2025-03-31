package tensor

import "log"

// Sub 张量减法
func (t *Tensor) Sub_bak(other *Tensor) *Tensor {
	defer func() {
		if err := recover(); err != nil {
			log.Println("t.Shape:", t.Shape, " other Shape:", other.Shape)
			panic(err)
		}
	}()
	// 需要实现张量形状检查
	result := make([]float64, len(t.Data))
	for i := range t.Data {
		result[i] = t.Data[i] - other.Data[i]
	}
	return &Tensor{Data: result, Shape: t.Shape}
}
