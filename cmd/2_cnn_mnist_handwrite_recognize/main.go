package main

import (
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_loader/mnist"
	"github.com/Jimmy2099/torch/data_struct/matrix"
	"log"
	"math/rand"
	"time"
)

// CNN 定义简单的卷积神经网络结构
type CNN struct {
	conv1 *torch.ConvLayer
	pool1 *torch.MaxPoolLayer
	fc1   *torch.LinearLayer
	fc2   *torch.LinearLayer
}

func (c *CNN) Parameters() []*matrix.Matrix {
	//TODO implement me
	panic("implement me")
}

func NewCNN() *CNN {
	rand.Seed(time.Now().UnixNano())
	return &CNN{
		conv1: torch.NewConvLayer(1, 8, 3, 2, 1), // 输入通道1，输出通道8，卷积核3x3
		pool1: torch.NewMaxPoolLayer(2, 2),       // 2x2最大池化
		fc1:   torch.NewLinearLayer(8*12*12, 64), // 全连接层1
		fc2:   torch.NewLinearLayer(64, 10),      // 全连接层2，输出10个类别
	}
}

func (c *CNN) Forward(x *matrix.Matrix) *matrix.Matrix {
	// 前向传播
	x = c.conv1.Forward(x)
	x = c.pool1.Forward(x)
	x = x.Flatten()
	x = c.fc1.Forward(x)
	x = x.ReLU()
	x = c.fc2.Forward(x)
	return x.Softmax()
}

func (c *CNN) Backward(targets *matrix.Matrix, lr float64) {
	// 反向传播
	grad := c.fc2.Backward(targets, lr)
	grad = c.fc1.Backward(grad, lr)
	grad = grad.Reshape(8, 12)
	grad = c.pool1.Backward(grad)
	_ = c.conv1.BackwardWithLR(grad, lr)
}

func (c *CNN) ZeroGrad() {
	// 清零梯度
	c.conv1.ZeroGrad()
	c.fc1.ZeroGrad()
	c.fc2.ZeroGrad()
}

func main() {
	// 加载MNIST数据集
	trainData, err := mnist.LoadMNIST("./dataset/MNIST/raw/train-images-idx3-ubyte", "./dataset/MNIST/raw/train-labels-idx1-ubyte")
	if err != nil {
		log.Fatal(err)
	}
	X_train := trainData.Images
	Y_train := trainData.Labels

	// 创建CNN模型
	model := NewCNN()
	trainer := torch.NewBasicTrainer(CrossEntropyLoss)

	// 训练模型
	trainer.Train(model, X_train, Y_train, 10, 0.01)

	// 测试模型
	testData, err := mnist.LoadMNIST("./dataset/MNIST/raw/t10k-images-idx3-ubyte", "./dataset/MNIST/raw/t10k-labels-idx1-ubyte")
	if err != nil {
		log.Fatal(err)
	}
	X_test := testData.Images
	Y_test := testData.Labels

	// 计算测试集准确率
	accuracy := Evaluate(model, X_test, Y_test)
	fmt.Printf("Test Accuracy: %.2f%%\n", accuracy*100)
}

// CrossEntropyLoss 计算交叉熵损失
func CrossEntropyLoss(predictions, targets *matrix.Matrix) float64 {
	return 0
}

// Evaluate 计算模型准确率
func Evaluate(model *CNN, inputs, targets *matrix.Matrix) float64 {
	outputs := model.Forward(inputs)
	predictions := outputs.ArgMax()
	correct := 0
	for i := 0; i < predictions.Size(); i++ {
		if predictions.At(i, 0) == targets.At(i, 0) {
			correct++
		}
	}
	return float64(correct) / float64(predictions.Size())
}
