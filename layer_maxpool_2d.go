package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	math "github.com/chewxy/math32"
)

// MaxPool2DLayer 实现2D最大池化层
type MaxPool2DLayer struct {
	PoolSize  int
	Stride    int
	Padding   int
	Input     *tensor.Tensor // 保存输入用于反向传播
	ArgMax    [][4]int       // 记录最大值位置 [batch][channel][row][col]
	OutputDim []int          // 输出维度
}

// NewMaxPool2DLayer 创建新的最大池化层
func NewMaxPool2DLayer(poolSize, stride, padding int) *MaxPool2DLayer {
	if poolSize <= 0 || stride <= 0 {
		panic("pool size and stride must be positive")
	}
	return &MaxPool2DLayer{
		PoolSize: poolSize,
		Stride:   stride,
		Padding:  padding,
	}
}

// Forward 前向传播
func (m *MaxPool2DLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x == nil {
		panic("input tensor cannot be nil")
	}
	if len(x.Shape) != 4 {
		panic("MaxPool2D expects 4D input tensor [batch, channels, height, width]")
	}

	// 保存输入
	m.Input = x.Clone()

	// 计算输出维度
	batchSize := x.Shape[0]
	channels := x.Shape[1]
	inHeight := x.Shape[2]
	inWidth := x.Shape[3]

	outHeight := (inHeight+2*m.Padding-m.PoolSize)/m.Stride + 1
	outWidth := (inWidth+2*m.Padding-m.PoolSize)/m.Stride + 1

	m.OutputDim = []int{batchSize, channels, outHeight, outWidth}
	outputData := make([]float32, batchSize*channels*outHeight*outWidth)
	m.ArgMax = make([][4]int, len(outputData))

	// 执行池化操作
	for b := 0; b < batchSize; b++ {
		for c := 0; c < channels; c++ {
			for h := 0; h < outHeight; h++ {
				for w := 0; w < outWidth; w++ {
					// 计算输入区域
					hStart := h*m.Stride - m.Padding
					wStart := w*m.Stride - m.Padding
					hEnd := hStart + m.PoolSize
					wEnd := wStart + m.PoolSize

					// 边界检查
					hStart = max(0, hStart)
					wStart = max(0, wStart)
					hEnd = min(inHeight, hEnd)
					wEnd = min(inWidth, wEnd)

					// 查找最大值
					//TODO
					maxVal := float32(-math.MaxFloat32)
					maxH, maxW := 0, 0
					for i := hStart; i < hEnd; i++ {
						for j := wStart; j < wEnd; j++ {
							val := x.Get([]int{b, c, i, j})
							if val > maxVal {
								maxVal = float32(val)
								maxH, maxW = i, j
							}
						}
					}

					// 保存结果
					outIdx := b*channels*outHeight*outWidth + c*outHeight*outWidth + h*outWidth + w
					outputData[outIdx] = maxVal
					m.ArgMax[outIdx] = [4]int{b, c, maxH, maxW}
				}
			}
		}
	}

	return tensor.NewTensor(outputData, m.OutputDim)
}

// Backward 反向传播
func (m *MaxPool2DLayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	if m.Input == nil {
		panic("must call Forward first")
	}
	if gradOutput == nil {
		panic("gradient tensor cannot be nil")
	}
	if len(gradOutput.Shape) != 4 {
		panic("gradOutput must be 4D tensor")
	}

	// 初始化梯度张量
	gradInput := tensor.NewTensor(make([]float32, len(m.Input.Data)), m.Input.Shape)

	// 将梯度传播到最大值位置
	for idx, pos := range m.ArgMax {
		b, c, h, w := pos[0], pos[1], pos[2], pos[3]
		gradInput.Set1([]int{b, c, h, w}, gradOutput.Data[idx])
	}

	return gradInput
}

// OutputShape 返回输出形状
func (m *MaxPool2DLayer) OutputShape() []int {
	return m.OutputDim
}

// helper functions
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
