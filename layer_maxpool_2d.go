package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	math "github.com/chewxy/math32"
)

type MaxPool2DLayer struct {
	PoolSize  int
	Stride    int
	Padding   int
	Input     *tensor.Tensor
	ArgMax    [][4]int
	OutputDim []int
}

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

func (m *MaxPool2DLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x == nil {
		panic("input tensor cannot be nil")
	}
	if len(x.GetShape()) != 4 {
		panic("MaxPool2D expects 4D input tensor [batch, channels, height, width]")
	}

	m.Input = x.Clone()

	batchSize := x.GetShape()[0]
	channels := x.GetShape()[1]
	inHeight := x.GetShape()[2]
	inWidth := x.GetShape()[3]

	outHeight := (inHeight+2*m.Padding-m.PoolSize)/m.Stride + 1
	outWidth := (inWidth+2*m.Padding-m.PoolSize)/m.Stride + 1

	m.OutputDim = []int{batchSize, channels, outHeight, outWidth}
	outputData := make([]float32, batchSize*channels*outHeight*outWidth)
	m.ArgMax = make([][4]int, len(outputData))

	for b := 0; b < batchSize; b++ {
		for c := 0; c < channels; c++ {
			for h := 0; h < outHeight; h++ {
				for w := 0; w < outWidth; w++ {
					hStart := h*m.Stride - m.Padding
					wStart := w*m.Stride - m.Padding
					hEnd := hStart + m.PoolSize
					wEnd := wStart + m.PoolSize

					hStart = max(0, hStart)
					wStart = max(0, wStart)
					hEnd = min(inHeight, hEnd)
					wEnd = min(inWidth, wEnd)

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

					outIdx := b*channels*outHeight*outWidth + c*outHeight*outWidth + h*outWidth + w
					outputData[outIdx] = maxVal
					m.ArgMax[outIdx] = [4]int{b, c, maxH, maxW}
				}
			}
		}
	}

	return tensor.NewTensor(outputData, m.OutputDim)
}

func (m *MaxPool2DLayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	return nil
}

func (m *MaxPool2DLayer) OutputShape() []int {
	return m.OutputDim
}

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
