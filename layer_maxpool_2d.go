package torch

import (
	"github.com/Jimmy2099/torch/data_struct/matrix"
)

// MaxPoolLayer 最大池化层实现
type MaxPoolLayer struct {
	poolSize int
	stride   int
	padding  int // 添加padding字段
	// 添加缓存用于反向传播
	lastInput *matrix.Matrix
	argMax    [][2]int // 记录最大值的坐标
}

// 修改构造函数以支持padding参数
func NewMaxPool2DLayer(kSize, stride, padding int) *MaxPoolLayer {
	if kSize <= 0 || stride <= 0 {
		panic("pool size and stride must be positive")
	}
	return &MaxPoolLayer{
		poolSize: kSize,
		stride:   stride,
		padding:  padding,
	}
}

func (m *MaxPoolLayer) Forward(x *matrix.Matrix) *matrix.Matrix {
	if x == nil {
		panic("input matrix cannot be nil")
	}

	// 对输入进行padding（假设你有实现Pad方法）
	padded := x
	if m.padding > 0 {
		padded = x.Pad(m.padding)
	}

	// 保存输入用于反向传播
	m.lastInput = padded.Clone()

	// 执行最大池化并记录最大值位置
	output, argMax := padded.MaxPoolWithArgMax(m.poolSize, m.stride)

	// 将argMax转换为[][2]int格式
	m.argMax = make([][2]int, len(argMax))
	for i, idx := range argMax {
		row := idx / padded.Cols
		col := idx % padded.Cols
		m.argMax[i] = [2]int{row, col}
	}

	return output
}

func (m *MaxPoolLayer) Backward(dout *matrix.Matrix) *matrix.Matrix {
	if m.lastInput == nil {
		panic("must call Forward first")
	}
	if dout == nil {
		panic("gradient matrix cannot be nil")
	}

	// 创建与输入相同大小的梯度矩阵
	dx := matrix.NewMatrix(m.lastInput.Rows, m.lastInput.Cols)

	// 将梯度传播到最大值位置
	// 注意这里需要确保索引对应正确，因为输入可能经过了padding
	for i := 0; i < len(m.argMax); i++ {
		row := m.argMax[i][0]
		col := m.argMax[i][1]
		// 这里假设 dout 的排列顺序与 argMax 顺序一致
		dx.Data[row][col] = dout.Data[i/dout.Cols][i%dout.Cols]
	}

	// 如果前向过程进行了padding，则需要将梯度移除padding部分
	if m.padding > 0 {
		dx = dx.Crop(m.padding)
	}

	return dx
}
