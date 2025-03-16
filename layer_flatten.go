package torch

import "github.com/Jimmy2099/torch/data_struct/matrix"

// FlattenLayer 展平层
type FlattenLayer struct {
	inputShape []int // 保存输入形状以便反向传播时恢复
}

func NewFlattenLayer() *FlattenLayer { return &FlattenLayer{} }

func (f *FlattenLayer) Forward(x *matrix.Matrix) *matrix.Matrix {
	// 保存输入形状
	f.inputShape = []int{x.Rows, x.Cols}

	// 展平操作：将矩阵展平为二维矩阵 (rows, cols*1*1*1)
	return x.Reshape(x.Rows, x.Cols)
}

func (f *FlattenLayer) Backward(dout *matrix.Matrix) *matrix.Matrix {
	// 将梯度恢复成原始形状
	if len(f.inputShape) < 2 {
		panic("FlattenLayer: input shape not recorded during forward pass")
	}
	return dout.Reshape(f.inputShape[0], f.inputShape[1])
}
