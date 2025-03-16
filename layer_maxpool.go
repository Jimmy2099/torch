package torch

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/matrix"
)

// MaxPoolLayer 最大池化层
type MaxPoolLayer struct {
	poolSize int
	stride   int
	// 添加缓存用于反向传播
	lastInput *matrix.Matrix
	argMax    [][2]int // 记录最大值的坐标
}

func NewMaxPoolLayer(size, stride int) *MaxPoolLayer {
	if size <= 0 || stride <= 0 {
		panic(fmt.Sprint("pool size and stride must be positive"))
	}
	return &MaxPoolLayer{
		poolSize: size,
		stride:   stride,
	}
}

func (m *MaxPoolLayer) Forward(x *matrix.Matrix) *matrix.Matrix {
	if x == nil {
		panic("input matrix cannot be nil")
	}

	// 保存输入用于反向传播
	m.lastInput = x.Clone()

	// 执行最大池化并记录最大值位置
	output, argMax := x.MaxPoolWithArgMax(m.poolSize, m.stride)

	// 将argMax转换为[][2]int格式
	m.argMax = make([][2]int, len(argMax))
	for i, idx := range argMax {
		row := idx / x.Cols
		col := idx % x.Cols
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
	for i := 0; i < len(m.argMax); i++ {
		row := m.argMax[i][0]
		col := m.argMax[i][1]
		dx.Data[row][col] = dout.Data[i/dout.Cols][i%dout.Cols]
	}

	return dx
}
