package main

import (
	"encoding/csv"
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/matrix"
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
	pool  *torch.MaxPoolLayer
}

func (c *CNN) Parameters() []*matrix.Matrix {
	//TODO implement me
	panic("implement me")
}

func NewCNN() *CNN {
	rand.Seed(time.Now().UnixNano())
	cnn := &CNN{
		conv1: torch.NewConvLayer(1, 32, 3, 3, 1),
		conv2: torch.NewConvLayer(32, 64, 3, 1, 1),
		fc1:   torch.NewLinearLayer(64*7*7, 128),
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

	}
	return cnn
}

func (c *CNN) Forward(x *matrix.Matrix) *matrix.Matrix {
	// 前向传播
	//x = self.relu(x.conv1(x))
	//x = self.pool(x)

	//x = self.relu(self.conv2(x))
	//x = self.pool(x)

	//x = torch.flatten(x, 1)
	//x = self.relu(self.fc1(x))
	//x = self.fc2(x)

	x = c.conv1.Forward(x)
	x = c.relu.Forward(x)

	x = c.conv2.Forward(x)
	x = c.relu.Forward(x)
	x = c.pool.Forward(x)

	x = x.Flatten()

	x = c.fc1.Forward(x)
	x = c.relu.Forward(x)
	x = c.fc2.Forward(x)

	return x
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
		fmt.Println(model.Forward(images[0]))
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

// ReadCSV reads an image from a CSV file and converts it to a matrix
func ReadCSV(filepath string) (*matrix.Matrix, error) {
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

	// Create a 28x28 matrix from the flattened image data
	mat := matrix.NewMatrixFromSlice1D(data, 28, 28)
	//matrix(data)

	// Normalize pixel values to [0, 1] range (MNIST images are 0-255)
	mat.DivScalar(255.0) // Normalize image pixels to [0,1]

	return mat, nil
}

// LoadDataFromCSVDir loads image data from a directory of CSV files and returns matrices and labels
func LoadDataFromCSVDir(directory string) ([]*matrix.Matrix, []string, error) {
	var matrices []*matrix.Matrix
	var labels []string

	// Open the labels CSV file (assuming it's named 'labels.csv' and has the format: filename,label)
	labelFilePath := filepath.Join(directory, "labels.csv")
	labelFile, err := os.Open(labelFilePath)
	if err != nil {
		return nil, nil, fmt.Errorf("could not open labels file: %v", err)
	}
	defer labelFile.Close()

	// Read labels CSV file
	labelReader := csv.NewReader(labelFile)
	labelLines, err := labelReader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("could not read labels CSV: %v", err)
	}

	// Create a map of filenames to labels (for easy access)
	labelMap := make(map[string]string)
	for _, line := range labelLines {
		if len(line) < 2 {
			continue
		}
		labelMap[line[0]] = line[1]
	}

	// Iterate over all CSV image files in the directory
	err = filepath.Walk(directory, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories and ensure we're dealing with CSV files (except the labels CSV)
		if info.IsDir() || strings.HasSuffix(path, "labels.csv") {
			return nil
		}

		// Read the image CSV file
		image, err := ReadCSV(path)
		if err != nil {
			log.Printf("Error reading image CSV %s: %v", path, err)
			return nil // Skip this image and continue with the next one
		}

		// Extract the image filename (assume the file name is the same as the CSV name)
		_, filename := filepath.Split(path)
		label, ok := labelMap[filename]
		if !ok {
			label = "unknown" // If no label found, mark it as unknown
		}

		// Add the matrix and label to the slices
		matrices = append(matrices, image)
		labels = append(labels, label)

		return nil
	})

	if err != nil {
		return nil, nil, fmt.Errorf("error walking through directory: %v", err)
	}

	return matrices, labels, nil
}
