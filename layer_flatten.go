package torch

import "github.com/Jimmy2099/torch/data_struct/tensor"

// FlattenLayer 展平层实现
type FlattenLayer struct {
	inputShape []int // 保存输入形状以便反向传播时恢复
}

func (r *FlattenLayer) GetWeights() *tensor.Tensor {
	return &tensor.Tensor{
		Data:  make([]float32, 0),
		Shape: make([]int, 0),
	}
}

func (r *FlattenLayer) GetBias() *tensor.Tensor {
	return &tensor.Tensor{
		Data:  make([]float32, 0),
		Shape: make([]int, 0),
	}
}

// NewFlattenLayer 创建新的展平层
func NewFlattenLayer() *FlattenLayer {
	return &FlattenLayer{}
}

// Forward 前向传播
func (f *FlattenLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	// 保存输入形状
	f.inputShape = make([]int, len(x.Shape))
	copy(f.inputShape, x.Shape)

	// 计算展平后的元素总数
	totalElements := 1
	for _, dim := range x.Shape {
		totalElements *= dim
	}

	// 展平操作：将输入张量展平为一维
	return x.Reshape([]int{totalElements})
}

// Backward 反向传播
func (f *FlattenLayer) Backward(dout *tensor.Tensor) *tensor.Tensor {
	// 将梯度恢复成原始形状
	if len(f.inputShape) == 0 {
		panic("FlattenLayer: input shape not recorded during forward pass")
	}
	return dout.Reshape(f.inputShape)
}
