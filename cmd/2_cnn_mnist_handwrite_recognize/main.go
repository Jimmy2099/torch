package main

import (
	"encoding/csv"
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/matrix"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// CNN 定义简单的卷积神经网络结构
type CNN struct {
	//        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
	//        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
	//        self.fc1 = torch.nn.Linear(in_features=64 * 7 * 7, out_features=128)
	//        self.fc2 = torch.nn.Linear(in_features=128, out_features=10)
	//        self.relu = torch.nn.ReLU()
	//        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
	conv1 *torch.ConvLayer
	conv2 *torch.ConvLayer
	fc1   *torch.LinearLayer
	fc2   *torch.LinearLayer
	relu  *torch.ReLULayer
	pool  *torch.MaxPool2DLayer
}

func (c *CNN) Parameters() []*matrix.Matrix {
	//TODO implement me
	panic("implement me")
}

// Forward performs the forward pass of the CNN.
func (c *CNN) Forward(x *tensor.Tensor) *tensor.Tensor {
	fmt.Println("\n=== Starting Forward Pass ===")
	fmt.Printf("Input shape: %v\n", x.Shape)

	// Conv1: (1,1,28,28) -> (1,32,28,28)
	fmt.Println("\nConv1:")
	x = c.conv1.Forward(x)
	fmt.Printf("After conv1: %v\n", x.Shape)

	// ReLU1: (1,32,28,28) -> (1,32,28,28)
	fmt.Println("\nReLU1:")
	x = c.relu.Forward(x)
	fmt.Printf("After relu1: %v\n", x.Shape)

	// Pool1: (1,32,28,28) -> (1,32,14,14)
	fmt.Println("\nPool1:")
	x = c.pool.Forward(x)
	fmt.Printf("After pool1: %v\n", x.Shape)

	// Conv2: (1,32,14,14) -> (1,64,14,14)
	fmt.Println("\nConv2:")
	x = c.conv2.Forward(x)
	fmt.Printf("After conv2: %v\n", x.Shape)

	// ReLU2: (1,64,14,14) -> (1,64,14,14)
	fmt.Println("\nReLU2:")
	x = c.relu.Forward(x)
	fmt.Printf("After relu2: %v\n", x.Shape)

	// Pool2: (1,64,14,14) -> (1,64,7,7)
	fmt.Println("\nPool2:")
	x = c.pool.Forward(x)
	fmt.Printf("After pool2: %v\n", x.Shape)

	// Flatten: (1,64,7,7) -> (1,3136)
	fmt.Println("\nFlatten:")
	x = x.Flatten()
	fmt.Printf("After flatten: %v\n", x.Shape)

	// FC1: (1,3136) -> (1,128)
	fmt.Println("\nFC1:")
	x = c.fc1.Forward(x)
	fmt.Printf("After fc1: %v\n", x.Shape)

	// ReLU3: (1,128) -> (1,128)
	fmt.Println("\nReLU3:")
	x = c.relu.Forward(x)
	fmt.Printf("After relu3: %v\n", x.Shape)

	// FC2: (1,128) -> (1,10)
	fmt.Println("\nFC2:")
	x = c.fc2.Forward(x)
	fmt.Printf("After fc2: %v\n", x.Shape)

	fmt.Println("\n=== Forward Pass Complete ===")
	return x
}

func NewCNN() *CNN {
	rand.Seed(time.Now().UnixNano())
	cnn := &CNN{
		conv1: torch.NewConvLayer(1, 32, 3, 1, 1),  // in=1, out=32, kernel=3, stride=1, pad=1
		conv2: torch.NewConvLayer(32, 64, 3, 1, 1), // in=32, out=64, kernel=3, stride=1, pad=1
		fc1:   torch.NewLinearLayer(64*7*7, 128),   // 64*7*7=3136
		fc2:   torch.NewLinearLayer(128, 10),
		relu:  torch.NewReLULayer(),
		pool:  torch.NewMaxPool2DLayer(2, 2, 0),
	}

	//cnn.LoadModelFromCSV()
	layer := []string{
		"conv1",
		"conv2",
		"fc1",
		"fc2",
	}
	{
		num := 0
		d, _ := os.Getwd()
		weightData, err := torch.LoadMatrixFromCSV(filepath.Join(d, "py", "data", layer[num]+".weight.csv"))
		if err != nil {
			panic(err)
		}
		cnn.conv1.SetWeights(weightData.Data)
		biasData, err := torch.LoadMatrixFromCSV(filepath.Join(d, "py", "data", layer[num]+".bias.csv"))
		if err != nil {
			panic(err)
		}
		cnn.conv1.SetBias(biasData.Data)
		cnn.conv1.Weights.Reshape([]int{32, 1, 3, 3})
		cnn.conv1.Bias.Reshape([]int{32})
	}
	{
		num := 1
		d, _ := os.Getwd()
		weightData, err := torch.LoadMatrixFromCSV(filepath.Join(d, "py", "data", layer[num]+".weight.csv"))
		if err != nil {
			panic(err)
		}
		cnn.conv2.SetWeights(weightData.Data)
		biasData, err := torch.LoadMatrixFromCSV(filepath.Join(d, "py", "data", layer[num]+".bias.csv"))
		if err != nil {
			panic(err)
		}
		cnn.conv2.SetBias(biasData.Data)
		cnn.conv2.Weights.Reshape([]int{64, 32, 3, 3})
		cnn.conv2.Bias.Reshape([]int{64})
	}
	{
		num := 2
		d, _ := os.Getwd()
		weightData, err := torch.LoadMatrixFromCSV(filepath.Join(d, "py", "data", layer[num]+".weight.csv"))
		if err != nil {
			panic(err)
		}
		cnn.fc1.SetWeights(weightData.Data)
		biasData, err := torch.LoadMatrixFromCSV(filepath.Join(d, "py", "data", layer[num]+".bias.csv"))
		if err != nil {
			panic(err)
		}
		cnn.fc1.SetBias(biasData.Data)
		cnn.fc1.Weights.Reshape([]int{128, 3136})
		cnn.fc1.Bias.Reshape([]int{128})
	}
	{
		num := 3
		d, _ := os.Getwd()
		weightData, err := torch.LoadMatrixFromCSV(filepath.Join(d, "py", "data", layer[num]+".weight.csv"))
		if err != nil {
			panic(err)
		}
		cnn.fc2.SetWeights(weightData.Data)
		biasData, err := torch.LoadMatrixFromCSV(filepath.Join(d, "py", "data", layer[num]+".bias.csv"))
		if err != nil {
			panic(err)
		}
		cnn.fc2.SetBias(biasData.Data)
		cnn.fc2.Weights.Reshape([]int{10, 128})
		cnn.fc2.Bias.Reshape([]int{10})
	}
	return cnn
}

// TODO Backward
func (c *CNN) Backward(targets *matrix.Matrix, lr float64) {
	//// 反向传播
	//grad := c.fc2.Backward(targets, lr)
	//grad = c.fc1.Backward(grad, lr)
	//grad = grad.Reshape(8, 12)
	//grad = c.pool1.Backward(grad)
	//_ = c.conv1.BackwardWithLR(grad, lr)
}

func (c *CNN) ZeroGrad() {
	// 清零梯度
	c.conv1.ZeroGrad()
	c.fc1.ZeroGrad()
	c.fc2.ZeroGrad()
}

func main() {
	//// 加载MNIST数据集
	//trainData, err := mnist.LoadMNIST("./dataset/MNIST/raw/train-images-idx3-ubyte", "./dataset/MNIST/raw/train-labels-idx1-ubyte")
	//if err != nil {
	//	log.Fatal(err)
	//}
	//X_train := trainData.Images
	//Y_train := trainData.Labels

	// 创建CNN模型

	model := NewCNN()
	//trainer := torch.NewBasicTrainer(CrossEntropyLoss)

	//// 训练模型
	//trainer.Train(model, X_train, Y_train, 10, 0.01)
	//
	// 测试模型
	//testData, err := mnist.LoadMNIST("./dataset/MNIST/raw/t10k-images-idx3-ubyte", "./dataset/MNIST/raw/t10k-labels-idx1-ubyte")
	//if err != nil {
	//	log.Fatal(err)
	//}
	//X_test := testData.Images
	//Y_test := testData.Labels

	{
		directory := "./py/mnist_images" // Adjust the path to where your image CSVs and labels.csv are stored

		// Load data (images and labels)
		images, labels, err := LoadDataFromCSVDir(directory)
		if err != nil {
			log.Fatalf("Error loading data: %v", err)
		}

		// Example: Print the first image matrix and its corresponding label
		if len(images) > 0 && len(labels) > 0 {
			fmt.Printf("First Image Matrix:\n%v\n", images[0])
			fmt.Printf("First Image Label: %s\n", labels[0])
		}

		num := 4
		prediction := Predict(model, images[num])
		fmt.Println("Label:", labels[num], "prediction:", prediction.Data[0][0])
	}

	// 计算测试集准确率
	//accuracy := Evaluate(model, X_test, Y_test)
	//fmt.Printf("Test Accuracy: %.2f%%\n", accuracy*100)
}

// CrossEntropyLoss 计算交叉熵损失
func CrossEntropyLoss(predictions, targets *matrix.Matrix) float64 {
	return 0
}

// Evaluate 计算模型准确率
func Evaluate(model *CNN, inputs, targets *matrix.Matrix) float64 {
	//outputs := model.Forward(inputs)
	//predictions := outputs.ArgMax()
	//correct := 0
	//for i := 0; i < predictions.Size(); i++ {
	//	if predictions.At(i, 0) == targets.At(i, 0) {
	//		correct++
	//	}
	//}
	return 0 //float64(correct) / float64(predictions.Size())
}

// ReadCSV reads an image from a CSV file and converts it to a matrix
func ReadCSV(filepath string) (*tensor.Tensor, error) {
	// Open the CSV file
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read the CSV file
	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	// Flattened data of the image (784 pixels for MNIST)
	// We assume each line is a single pixel in the image, and the total number of values should be 28*28 = 784
	rows := len(lines)
	cols := len(lines[0])
	if rows*cols != 784 { // Ensure it matches the MNIST image size
		return nil, fmt.Errorf("expected 784 values in CSV, got %d", rows*cols)
	}

	// Convert the CSV data to a flat matrix (784,)
	data := make([]float64, rows*cols)

	for i, line := range lines {
		for j, value := range line {
			// Convert string to float64
			val, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
			data[i*cols+j] = val
		}
	}

	// Create a Tensor from the flattened image data
	tensorImage := tensor.NewTensor(data, []int{28, 28})

	return tensorImage, nil
}

func LoadDataFromCSVDir(directory string) ([]*tensor.Tensor, []string, error) {
	var tensors []*tensor.Tensor
	var labels []string

	// 读取标签文件
	labelFilePath := filepath.Join(directory, "labels.csv")
	labelFile, err := os.Open(labelFilePath)
	if err != nil {
		return nil, nil, fmt.Errorf("无法打开标签文件: %v", err)
	}
	defer labelFile.Close()

	reader := csv.NewReader(labelFile)
	labelRecords, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("读取标签CSV失败: %v", err)
	}

	// 创建文件名到标签的映射
	labelMap := make(map[string]string)
	for _, record := range labelRecords {
		if len(record) >= 2 {
			filename := strings.TrimSpace(record[0])
			label := strings.TrimSpace(record[1])
			labelMap[filename] = label
		}
	}

	// 读取图像文件
	files, err := os.ReadDir(directory)
	if err != nil {
		return nil, nil, fmt.Errorf("读取目录失败: %v", err)
	}

	for _, file := range files {
		if file.IsDir() || file.Name() == "labels.csv" {
			continue
		}
		if !strings.Contains(file.Name(), "png.csv") {
			continue
		}

		imagePath := filepath.Join(directory, file.Name())
		image, err := ReadCSV(imagePath)
		if err != nil {
			log.Printf("读取图像文件 %s 失败: %v", file.Name(), err)
			continue
		}

		// 获取对应的标签
		label := file.Name()

		tensors = append(tensors, image)
		labels = append(labels, label)
	}

	return tensors, labels, nil
}

func Predict(model *CNN, image *tensor.Tensor) *matrix.Matrix {
	// Pass the image through the model's forward pass
	image = image.Reshape([]int{1, 1, 28, 28})
	output := model.Forward(image)

	// Convert tensor to matrix for argmax operation
	outputMatrix := &matrix.Matrix{
		Data: [][]float64{output.Data},
		Rows: 1,
		Cols: len(output.Data),
	}

	return outputMatrix.ArgMax()
}
