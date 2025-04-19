package main

import (
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"math/rand"
	"time"
)

type NeuralNetwork struct {
	Layers []torch.Layer
}

func (nn *NeuralNetwork) Parameters() []*tensor.Tensor {
	params := make([]*tensor.Tensor, 0)
	for _, layer := range nn.Layers {
		if linearLayer, ok := layer.(*torch.LinearLayer); ok {
			params = append(params, linearLayer.Weights)
			params = append(params, linearLayer.Bias)
		}
	}
	return params
}

func (nn *NeuralNetwork) ZeroGrad() {
	for _, layer := range nn.Layers {
		if linearLayer, ok := layer.(*torch.LinearLayer); ok {
			linearLayer.ZeroGrad()
		}
	}
}

func NewNeuralNetwork(layerDims []int) *NeuralNetwork {
	return &NeuralNetwork{
		Layers: createLayers(layerDims),
	}
}

func createLayers(layerDims []int) []torch.Layer {
	layers := make([]torch.Layer, 0)
	for i := 0; i < len(layerDims)-1; i++ {
		layers = append(layers, torch.NewLinearLayer(layerDims[i], layerDims[i+1]))
		if i < len(layerDims)-2 {
			layers = append(layers, torch.NewReLULayer())
		}
	}
	return layers
}

func (nn *NeuralNetwork) Forward(x *tensor.Tensor) *tensor.Tensor {
	output := x
	for _, layer := range nn.Layers {
		output = layer.Forward(output)
	}
	return output
}

func (nn *NeuralNetwork) Backward(targets *tensor.Tensor, lr float32) {
	lastLayer := nn.Layers[len(nn.Layers)-1].(*torch.LinearLayer)
	output := lastLayer.Output

	if !tensor.ShapeEqual(output.GetShape(), targets.GetShape()) {
		targets = targets.Reshape(output.GetShape())
	}

	gradOutput := output.Sub(targets)
	gradOutput = gradOutput.MulScalar(2.0 / float32(targets.GetShape()[0]))

	for i := len(nn.Layers) - 1; i >= 0; i-- {
		gradOutput = nn.Layers[i].Backward(gradOutput, lr)
	}
}

func polynomialFeatures(X *tensor.Tensor, degree int) *tensor.Tensor {
	numSamples, numFeatures := X.GetShape()[0], X.GetShape()[1]
	newFeatures := numFeatures * degree
	features := make([]float32, numSamples*newFeatures)

	for s := 0; s < numSamples; s++ {
		for f := 0; f < numFeatures; f++ {
			val := X.Data[s*numFeatures+f]
			for d := 1; d <= degree; d++ {
				features[s*newFeatures+f*degree+(d-1)] = math.Pow(val, float32(d))
			}
		}
	}
	return tensor.NewTensor(features, []int{numSamples, newFeatures})
}

func targetFunc(x1, x2 float32) float32 {
	return 3 + 2*x1 + 1.5*x2 + 0.5*x1*x1 - 0.8*x2*x2 + 0.3*x1*x1*x1
}

func main() {
	rand.Seed(time.Now().UnixNano())

	X_data := make([]float32, 10*2)
	for i := range X_data {
		X_data[i] = float32(rand.Float32())
	}
	X_train := tensor.NewTensor(X_data, []int{10, 2})

	y_data := make([]float32, 10)
	for j := 0; j < 10; j++ {
		x1 := X_train.Data[j*2]
		x2 := X_train.Data[j*2+1]
		y_data[j] = targetFunc(x1, x2)
	}
	y_train := tensor.NewTensor(y_data, []int{10, 1})

	degree := 3
	X_train_poly := polynomialFeatures(X_train, degree)
	fmt.Printf("Training data shape: %v\n", X_train_poly.GetShape())

	model := NewNeuralNetwork([]int{6, 10, 1})

	fmt.Println("\n=== shape Verification ===")
	fmt.Printf("Input shape: %v\n", X_train_poly.GetShape())
	fmt.Printf("Target shape: %v\n", y_train.GetShape())
	sampleOutput := model.Forward(X_train_poly)
	fmt.Printf("Model output shape: %v\n", sampleOutput.GetShape())

	trainer := torch.NewBasicTrainer(torch.MSE)
	epochs := 500
	learningRate := float32(0.0)

	trainer.Train(model, X_train_poly, y_train, epochs, float32(learningRate))

	test_data := []float32{0.2, 0.3}
	test_sample := tensor.NewTensor(test_data, []int{1, 2})
	test_poly := polynomialFeatures(test_sample, degree)
	prediction := model.Forward(test_poly)

	fmt.Printf("\nPrediction: %.4f\n", prediction.Data[0])
	fmt.Printf("Actual value: %.4f\n", targetFunc(test_data[0], test_data[1]))
	fmt.Printf("Error: %.4f\n", math.Abs(prediction.Data[0]-targetFunc(test_data[0], test_data[1])))
}
