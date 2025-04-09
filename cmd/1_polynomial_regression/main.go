package main

import (
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/tensor"
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

	// 确保输出和目标形状一致
	if !tensor.ShapeEqual(output.Shape, targets.Shape) {
		targets = targets.Reshape(output.Shape)
	}

	// 计算梯度 (output - targets) * 2/batch_size
	gradOutput := output.Sub(targets)
	gradOutput = gradOutput.MulScalar(2.0 / float32(targets.Shape[0]))

	// 反向传播
	for i := len(nn.Layers) - 1; i >= 0; i-- {
		gradOutput = nn.Layers[i].Backward(gradOutput, lr)
	}
}

// 修正后的多项式特征生成
func polynomialFeatures(X *tensor.Tensor, degree int) *tensor.Tensor {
	numSamples, numFeatures := X.Shape[0], X.Shape[1]
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

	// 生成训练数据 (样本数, 特征数) = (10, 2)
	X_data := make([]float32, 10*2)
	for i := range X_data {
		X_data[i] = float32(rand.Float32())
	}
	X_train := tensor.NewTensor(X_data, []int{10, 2}) // [样本数, 特征数]

	// 生成目标值 (样本数, 1)
	y_data := make([]float32, 10)
	for j := 0; j < 10; j++ {
		x1 := X_train.Data[j*2]
		x2 := X_train.Data[j*2+1]
		y_data[j] = targetFunc(x1, x2)
	}
	y_train := tensor.NewTensor(y_data, []int{10, 1})

	// 生成3次多项式特征 (10, 2*3=6)
	degree := 3
	X_train_poly := polynomialFeatures(X_train, degree)
	fmt.Printf("训练数据形状: %v\n", X_train_poly.Shape) // 应输出 [10 6]

	// 创建神经网络 [6输入 -> 10隐藏 -> 1输出]
	model := NewNeuralNetwork([]int{6, 10, 1})

	// 训练前验证形状
	fmt.Println("\n=== 形状验证 ===")
	fmt.Printf("输入形状: %v\n", X_train_poly.Shape) // [10 6]
	fmt.Printf("目标形状: %v\n", y_train.Shape)      // [10 1]
	sampleOutput := model.Forward(X_train_poly)
	fmt.Printf("模型输出形状: %v\n", sampleOutput.Shape) // 应输出 [10 1]

	// 训练配置
	trainer := torch.NewBasicTrainer(torch.MSE)
	epochs := 500
	learningRate := float32(0.0)

	// 开始训练
	trainer.Train(model, X_train_poly, y_train, epochs, float32(learningRate))

	// 测试预测
	test_data := []float32{0.2, 0.3}
	test_sample := tensor.NewTensor(test_data, []int{1, 2}) // [1样本, 2特征]
	test_poly := polynomialFeatures(test_sample, degree)    // [1, 6]
	prediction := model.Forward(test_poly)                  // [1, 1]

	fmt.Printf("\n预测值: %.4f\n", prediction.Data[0])
	fmt.Printf("真实值: %.4f\n", targetFunc(test_data[0], test_data[1]))
	fmt.Printf("误差: %.4f\n", math.Abs(prediction.Data[0]-targetFunc(test_data[0], test_data[1])))
}
