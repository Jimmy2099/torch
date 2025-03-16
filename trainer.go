package torch

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/matrix"
	"time"
)

// BasicTrainer 实现基本的训练器
type BasicTrainer struct {
	LossFunc func(predictions, targets *matrix.Matrix) float64
	Verbose  bool
}

func NewBasicTrainer(lossFunc func(predictions, targets *matrix.Matrix) float64) *BasicTrainer {
	return &BasicTrainer{
		LossFunc: lossFunc,
		Verbose:  true,
	}
}

func (t *BasicTrainer) Train(model Model, inputs, targets *matrix.Matrix, epochs int, learningRate float64) {
	start := time.Now()

	// 记录损失历史
	lossHistory := make([]float64, 0, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		// 前向传播
		outputs := model.Forward(inputs)

		// 计算损失
		loss := t.LossFunc(outputs, targets)
		lossHistory = append(lossHistory, loss)

		// 反向传播
		model.ZeroGrad()
		model.Backward(targets, learningRate)

		// 打印训练信息
		if t.Verbose && (epoch+1)%50 == 0 {
			fmt.Printf("Epoch [%d/%d], Loss: %.4f, Time: %v\n",
				epoch+1, epochs, loss, time.Since(start))
		}
	}

	// 绘制损失曲线
	if t.Verbose {
		printLoss(lossHistory)
	}
}

// plotLoss 绘制损失曲线
func printLoss(lossHistory []float64) {
	fmt.Println("\nTraining complete! Loss history:")
	for i, loss := range lossHistory {
		if i%50 == 0 {
			fmt.Printf("Epoch %d: %.4f\n", i, loss)
		}
	}
}

func (t *BasicTrainer) Validate(model Model, inputs, targets *matrix.Matrix) float64 {
	outputs := model.Forward(inputs)
	return t.LossFunc(outputs, targets)
}
