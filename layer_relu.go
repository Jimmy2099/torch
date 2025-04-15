package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

// ReLULayer 实现带泄漏的ReLU激活层
type ReLULayer struct {
	input    *tensor.Tensor // 保存输入用于反向传播
	negative float32        // 负半轴斜率（支持LeakyReLU）
	inplace  bool           // 是否原地操作
}

func (r *ReLULayer) GetWeights() *tensor.Tensor {
	return &tensor.Tensor{
		Data:  make([]float32, 0),
		Shape: make([]int, 0),
	}
}

func (r *ReLULayer) GetBias() *tensor.Tensor {
	return &tensor.Tensor{
		Data:  make([]float32, 0),
		Shape: make([]int, 0),
	}
}

// NewReLULayer 创建标准ReLU层（负半轴斜率为0）
func NewReLULayer() *ReLULayer {
	return &ReLULayer{
		negative: 0.0,
		inplace:  false,
	}
}

// NewLeakyReLULayer 创建带泄漏的ReLU层
func NewLeakyReLULayer(negativeSlope float32) *ReLULayer {
	if negativeSlope < 0 {
		panic("negative slope must be non-negative")
	}
	return &ReLULayer{
		negative: negativeSlope,
		inplace:  false,
	}
}

// SetInplace 设置是否原地操作（节省内存但会破坏输入数据）
func (r *ReLULayer) SetInplace(inplace bool) {
	r.inplace = inplace
}

// Forward 前向传播
func (r *ReLULayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x == nil {
		panic("input tensor cannot be nil")
	}

	// 保存输入（非原地操作时克隆）
	if r.inplace {
		r.input = x
	} else {
		r.input = x.Clone()
	}

	// 应用ReLU激活函数
	return x.Apply(func(val float32) float32 {
		if val > 0 {
			return val
		}
		return r.negative * val
	})
}

// Backward 反向传播
//func (r *ReLULayer) Backward(dout *tensor.Tensor) *tensor.Tensor {
//	if r.input == nil {
//		panic("must call Forward first")
//	}
//	if dout == nil {
//		panic("gradient tensor cannot be nil")
//	}
//	if !shapeEqual(r.input.Shape, dout.Shape) {
//		panic("input and gradient shapes must match")
//	}
//
//	// 计算ReLU梯度
//	grad := r.input.Apply(func(val float32) float32 {
//		if val > 0 {
//			return 1.0
//		}
//		return r.negative
//	})
//
//	// 与上游梯度相乘
//	return grad.Multiply(dout)
//}

// shapeEqual 辅助函数：比较形状是否相同
func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ActivationType 返回激活函数类型
func (r *ReLULayer) ActivationType() string {
	if r.negative == 0 {
		return "ReLU"
	}
	return "LeakyReLU"
}

// NegativeSlope 返回负半轴斜率
func (r *ReLULayer) NegativeSlope() float32 {
	return r.negative
}

func (r *ReLULayer) Backward(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor {
	if r.input == nil {
		panic("must call Forward first")
	}
	if gradOutput == nil {
		panic("gradient tensor cannot be nil")
	}
	if !shapeEqual(r.input.Shape, gradOutput.Shape) {
		panic("input and gradient shapes must match")
	}

	// 计算ReLU梯度
	grad := r.input.Apply(func(val float32) float32 {
		if val > 0 {
			return 1.0
		}
		return r.negative
	})

	// 与上游梯度相乘
	return grad.Multiply(gradOutput)
}

func (r *ReLULayer) ZeroGrad() {
	// ReLU层没有需要清零的梯度参数
}

func (r *ReLULayer) Parameters() []*tensor.Tensor {
	// ReLU层没有可训练参数
	return nil
}
