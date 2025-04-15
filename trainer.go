package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"time"
)

type BasicTrainer struct {
	LossFunc func(predictions, targets *tensor.Tensor) float32
	Verbose  bool
}

func NewBasicTrainer(lossFunc func(predictions, targets *tensor.Tensor) float32) *BasicTrainer {
	return &BasicTrainer{
		LossFunc: lossFunc,
		Verbose:  true,
	}
}

func (t *BasicTrainer) Train(model ModelInterface, inputs, targets *tensor.Tensor, epochs int, learningRate float32) {
	start := time.Now()

	lossHistory := make([]float32, 0, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		outputs := model.Forward(inputs)

		loss := t.LossFunc(outputs, targets)
		lossHistory = append(lossHistory, loss)

		model.ZeroGrad()
		model.Backward(targets, learningRate)

		if t.Verbose && (epoch+1)%50 == 0 {
			fmt.Printf("Epoch [%d/%d], Loss: %.4f, Time: %v\n",
				epoch+1, epochs, loss, time.Since(start))
		}
	}

	if t.Verbose {
		printLoss(lossHistory)
	}
}

func printLoss(lossHistory []float32) {
	fmt.Println("\nTraining complete! Loss history:")
	for i, loss := range lossHistory {
		if i%50 == 0 {
			fmt.Printf("Epoch %d: %.4f\n", i, loss)
		}
	}
}

func (t *BasicTrainer) Validate(model ModelInterface, inputs, targets *tensor.Tensor) float32 {
	outputs := model.Forward(inputs)
	return t.LossFunc(outputs, targets)
}
