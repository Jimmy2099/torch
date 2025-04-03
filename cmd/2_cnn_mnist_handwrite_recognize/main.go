package main

import (
	"encoding/csv"
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/matrix"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"github.com/Jimmy2099/torch/testing"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// CNN
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

	{
		//generate test data
		d, _ := os.Getwd()
		runCommand(filepath.Join(d, "py"), filepath.Join(d, "py", "generate_go_testdata.py"))
	}
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

	//num := 4
	result := []int{}
	for num := 0; num < len(images); num++ {
		model := NewCNN()
		prediction := Predict(model, images[num])
		fmt.Println("Label:", labels[num], "prediction:", prediction.Data[0][0])
		result = append(result, int(prediction.Data[0][0]))
	}
	fmt.Println("label:", labels)
	fmt.Println("predictions:", result)
	fileName := []string{}
	{

		for i := 0; i < len(labels); i++ {
			d, _ := os.Getwd()
			name := filepath.Join(d, "py", "mnist_images", labels[i])
			name = strings.Replace(name, ".csv", "", -1)
			fileName = append(fileName, name)
			fmt.Println()
		}
		if !testing.IsTesting() {
			predictPlot(fileName, result)
		}
	}

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

func predictPlot(imagePaths []string, predictions []int) error {
	if len(imagePaths) != len(predictions) {
		return fmt.Errorf("image paths and predictions length mismatch")
	}

	// 随机数后缀（保证每次文件名不同，不带 * 号）
	rand.Seed(time.Now().UnixNano())
	suffix := fmt.Sprintf("%d", rand.Intn(1000000))

	// 构造临时 Python 脚本文件名，例如：predict_plot_123456.py
	tmpScriptFileName := filepath.Join(os.TempDir(), "predict_plot_"+suffix+".py")
	// 构造临时数据文件名，例如：image_predictions_123456.txt
	tmpDataFileName := filepath.Join(os.TempDir(), "image_predictions_"+suffix+".txt")

	// Python 脚本内容，不包含任何数据，数据从数据文件传入
	pythonScript := `
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_data(file_path):
    image_paths = []
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            image_paths.append(parts[0])
            predictions.append(int(parts[1]))
    return image_paths, predictions

def predict_plot(data_file):
    image_paths, predictions = load_data(data_file)
    num = len(image_paths)
    fig, axes = plt.subplots(1, num, figsize=(15, 1.5))
    # 如果只有一张图，则将 axes 转换为列表
    if num == 1:
        axes = [axes]
    for i, (img_path, pred) in enumerate(zip(image_paths, predictions)):
        img = mpimg.imread(img_path)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Pred: {pred}', fontsize=10)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_plot.py <data_file_path>")
        sys.exit(1)
    data_file = sys.argv[1]
    predict_plot(data_file)
`
	// 创建 Python 脚本文件
	scriptFile, err := os.Create(tmpScriptFileName)
	if err != nil {
		return fmt.Errorf("unable to create temp Python script: %v", err)
	}
	_, err = scriptFile.WriteString(pythonScript)
	if err != nil {
		scriptFile.Close()
		return fmt.Errorf("unable to write Python script: %v", err)
	}
	scriptFile.Close()
	defer os.Remove(tmpScriptFileName) // 执行完后删除文件

	// 创建临时数据文件，写入图片路径和预测结果，每行格式：图片路径,预测结果
	dataFile, err := os.Create(tmpDataFileName)
	if err != nil {
		return fmt.Errorf("unable to create temp data file: %v", err)
	}
	for i := 0; i < len(imagePaths); i++ {
		_, err := dataFile.WriteString(fmt.Sprintf("%s,%d\n", imagePaths[i], predictions[i]))
		if err != nil {
			dataFile.Close()
			return fmt.Errorf("unable to write to temp data file: %v", err)
		}
	}
	dataFile.Close()
	defer os.Remove(tmpDataFileName)

	// 调用 Python 脚本，将数据文件路径作为参数传入
	cmd := exec.Command("python", tmpScriptFileName, tmpDataFileName)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("error executing Python script: %v", err)
	}

	return nil
}

func runCommand(workSpace string, fileName string) {

	// 执行 Python 脚本
	cmd := exec.Command("python", fileName)
	cmd.Dir = workSpace
	// 实时打印输出
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(fmt.Sprintln("Error creating Stdout pipe:", err))
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		panic(fmt.Sprintln("Error creating Stderr pipe:", err))
	}

	// 启动命令
	if err := cmd.Start(); err != nil {
		panic(fmt.Sprintln("Error starting Python script:", err))
	}

	// 使用 Goroutines 来实时打印标准输出和错误输出
	go func() {
		_, err := io.Copy(os.Stdout, stdout)
		if err != nil {
			panic(fmt.Sprintln("Error copying stdout:", err))
		}
	}()

	go func() {
		_, err := io.Copy(os.Stderr, stderr)
		if err != nil {
			panic(fmt.Sprintln("Error copying stderr:", err))
		}
	}()

	// 等待命令执行完成
	if err := cmd.Wait(); err != nil {
		panic(fmt.Sprintln("Error waiting for Python script to finish:", err))
	}
}
