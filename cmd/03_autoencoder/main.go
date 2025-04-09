// Filename: auto_encoder.go
package main

import (
	"encoding/csv"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/pkg/log"
	"github.com/Jimmy2099/torch/testing"
	"io"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// AutoEncoder defines the structure of the autoencoder model.
// Based on the Python structure:
// Autoencoder(
//
//	(encoder): Sequential(
//	  (0): Linear(in_features=784, out_features=128, bias=True) -> fc1
//	  (1): ReLU()                                            -> relu1
//	  (2): Linear(in_features=128, out_features=64, bias=True)  -> fc2
//	  (3): ReLU()                                            -> relu2
//	  (4): Linear(in_features=64, out_features=32, bias=True)   -> fc3
//	)
//	(decoder): Sequential(
//	  (0): Linear(in_features=32, out_features=64, bias=True)   -> fc4
//	  (1): ReLU()                                            -> relu3
//	  (2): Linear(in_features=64, out_features=128, bias=True)  -> fc5
//	  (3): ReLU()                                            -> relu4
//	  (4): Linear(in_features=128, out_features=784, bias=True) -> fc6
//	  (5): Sigmoid()                                         -> Sigmoid1
//	)
//
// )
// Layer: encoder.0.weight, Shape: torch.Size([128, 784]), dim: 2
// Layer: encoder.0.bias, Shape: torch.Size([128]), dim: 1
// Layer: encoder.2.weight, Shape: torch.Size([64, 128]), dim: 2
// Layer: encoder.2.bias, Shape: torch.Size([64]), dim: 1
// Layer: encoder.4.weight, Shape: torch.Size([32, 64]), dim: 2
// Layer: encoder.4.bias, Shape: torch.Size([32]), dim: 1
// Layer: decoder.0.weight, Shape: torch.Size([64, 32]), dim: 2
// Layer: decoder.0.bias, Shape: torch.Size([64]), dim: 1
// Layer: decoder.2.weight, Shape: torch.Size([128, 64]), dim: 2
// Layer: decoder.2.bias, Shape: torch.Size([128]), dim: 1
// Layer: decoder.4.weight, Shape: torch.Size([784, 128]), dim: 2
// Layer: decoder.4.bias, Shape: torch.Size([784]), dim: 1
type AutoEncoder struct {
	// Encoder layers
	fc1   *torch.LinearLayer
	relu1 *torch.ReLULayer
	fc2   *torch.LinearLayer
	relu2 *torch.ReLULayer
	fc3   *torch.LinearLayer // Latent space representation

	// Decoder layers
	fc4      *torch.LinearLayer
	relu3    *torch.ReLULayer
	fc5      *torch.LinearLayer
	relu4    *torch.ReLULayer
	fc6      *torch.LinearLayer
	Sigmoid1 *torch.SigmoidLayer // Output activation
}

// Helper structure to group layer loading information
type layerLoadInfo struct {
	pyTorchName string             // Name used in PyTorch state_dict keys (e.g., "encoder.0")
	goLayer     *torch.LinearLayer // Pointer to the corresponding layer in the Go struct
	weightShape []int              // Expected shape of the weight tensor
	biasShape   []int              // Expected shape of the bias tensor
}

// NewAutoEncoder creates and initializes a new AutoEncoder model.
// It loads the pre-trained weights and biases from CSV files using a pattern
// similar to the provided CNN example.
func NewAutoEncoder() *AutoEncoder {
	// 1. Initialize the AutoEncoder struct with new layers
	fmt.Println("Initializing AutoEncoder layers...")
	ae := &AutoEncoder{
		// Encoder
		fc1:   torch.NewLinearLayer(784, 128),
		relu1: torch.NewReLULayer(),
		fc2:   torch.NewLinearLayer(128, 64),
		relu2: torch.NewReLULayer(),
		fc3:   torch.NewLinearLayer(64, 32),

		// Decoder
		fc4:      torch.NewLinearLayer(32, 64),
		relu3:    torch.NewReLULayer(),
		fc5:      torch.NewLinearLayer(64, 128),
		relu4:    torch.NewReLULayer(),
		fc6:      torch.NewLinearLayer(128, 784),
		Sigmoid1: torch.NewSigmoidLayer(),
	}
	fmt.Println("Layers initialized.")

	// 2. Define the mapping between PyTorch names, Go layers, and shapes
	loadInfos := []layerLoadInfo{
		//Layer: encoder.0.weight, Shape: torch.Size([128, 784]), dim: 2
		//Layer: encoder.0.bias, Shape: torch.Size([128]), dim: 1
		{"encoder.0", ae.fc1, []int{128, 784}, []int{128}}, // fc1
		//Layer: encoder.2.weight, Shape: torch.Size([64, 128]), dim: 2
		//Layer: encoder.2.bias, Shape: torch.Size([64]), dim: 1
		{"encoder.2", ae.fc2, []int{64, 128}, []int{64}}, // fc2
		//Layer: encoder.4.weight, Shape: torch.Size([32, 64]), dim: 2
		//Layer: encoder.4.bias, Shape: torch.Size([32]), dim: 1
		{"encoder.4", ae.fc3, []int{32, 64}, []int{32}}, // fc3 (latent)
		//Layer: decoder.0.weight, Shape: torch.Size([64, 32]), dim: 2
		//Layer: decoder.0.bias, Shape: torch.Size([64]), dim: 1
		{"decoder.0", ae.fc4, []int{64, 32}, []int{64}}, // fc4
		//Layer: decoder.2.weight, Shape: torch.Size([128, 64]), dim: 2
		//Layer: decoder.2.bias, Shape: torch.Size([128]), dim: 1
		{"decoder.2", ae.fc5, []int{128, 64}, []int{128}}, // fc5
		//Layer: decoder.4.weight, Shape: torch.Size([784, 128]), dim: 2
		//Layer: decoder.4.bias, Shape: torch.Size([784]), dim: 1
		{"decoder.4", ae.fc6, []int{784, 128}, []int{784}}, // fc6
	}

	// 3. Get base data path
	basePath := "py/data" // Relative path to data directory from where Go runs
	d, err := os.Getwd()
	if err != nil {
		panic(fmt.Sprintf("Error getting working directory: %v", err))
	}
	dataPath := filepath.Join(d, basePath)
	fmt.Printf("Base data path set to: %s\n", dataPath)

	fmt.Println("\nLoading AutoEncoder model parameters...")

	// 4. Loop through the layer info and load parameters
	for _, info := range loadInfos {
		fmt.Printf("\n--- Loading parameters for: %s ---\n", info.pyTorchName)

		// --- Load Weights ---
		weightFileName := info.pyTorchName + ".weight.csv"
		weightFilePath := filepath.Join(dataPath, weightFileName)
		fmt.Printf("Attempting to load weights from: %s\n", weightFilePath)

		// Assuming LoadMatrixFromCSV returns a struct with a Data field (like [][]float32)
		weightMatrix, err := torch.LoadMatrixFromCSV(weightFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading weight file %s: %v", weightFilePath, err))
		}

		// Set weights using the method from the LinearLayer
		// Adapt this if the method signature is different (e.g., expects *tensor.Tensor)
		info.goLayer.SetWeights(weightMatrix.Data) // Assuming SetWeights takes [][]float32

		// Reshape the internal weight tensor *after* setting the data
		if info.goLayer.Weights == nil {
			panic(fmt.Sprintf("Weight tensor is nil after SetWeights for %s", info.pyTorchName))
		}
		info.goLayer.Weights.Reshape(info.weightShape) // Reshape internal tensor
		fmt.Printf("Loaded weights for %s, shape set to: %v\n", info.pyTorchName, info.goLayer.Weights.Shape)

		// --- Load Biases ---
		biasFileName := info.pyTorchName + ".bias.csv"
		biasFilePath := filepath.Join(dataPath, biasFileName)
		fmt.Printf("Attempting to load biases from: %s\n", biasFilePath)

		biasMatrix, err := torch.LoadMatrixFromCSV(biasFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading bias file %s: %v", biasFilePath, err))
		}

		// Set bias using the method from the LinearLayer
		info.goLayer.SetBias(biasMatrix.Data) // Assuming SetBias takes [][]float32

		// Reshape the internal bias tensor *after* setting the data
		if info.goLayer.Bias == nil {
			panic(fmt.Sprintf("Bias tensor is nil after SetBias for %s", info.pyTorchName))
		}
		info.goLayer.Bias.Reshape(info.biasShape) // Reshape internal tensor
		fmt.Printf("Loaded biases for %s, shape set to: %v\n", info.pyTorchName, info.goLayer.Bias.Shape)
	}

	fmt.Println("\n--- AutoEncoder model parameters loaded successfully. ---")

	// 5. Return the initialized and populated model
	return ae
}

// Forward performs the forward pass of the AutoEncoder.
func (ae *AutoEncoder) Forward(x *tensor.Tensor) *tensor.Tensor {
	// Input should be flattened image, e.g., shape [batch_size, 784] or [1, 784]
	if len(x.Shape) != 2 || x.Shape[1] != 784 {
		// Attempt to flatten if not already flat (e.g., if input is [1, 1, 28, 28])
		if x.Size() == 784 {
			fmt.Printf("Input shape %v is not [N, 784], attempting to flatten.\n", x.Shape)
			x = x.Flatten() // Flatten to [1, 784]
			fmt.Printf("Flattened input shape: %v\n", x.Shape)
		} else {
			panic(fmt.Sprintf("Input tensor shape %v is incompatible with AutoEncoder input (expected [N, 784])", x.Shape))
		}
	}

	fmt.Println("\n=== Starting AutoEncoder Forward Pass ===")
	fmt.Printf("Input shape: %v\n", x.Shape)

	// --- Encoder ---
	fmt.Println("\nEncoder FC1:")
	x = ae.fc1.Forward(x)
	fmt.Printf("After fc1: %v\n", x.Shape)

	fmt.Println("\nReLU1:")
	x = ae.relu1.Forward(x)
	fmt.Printf("After relu1: %v\n", x.Shape)

	fmt.Println("\nEncoder FC2:")
	x = ae.fc2.Forward(x)
	fmt.Printf("After fc2: %v\n", x.Shape)

	fmt.Println("\nReLU2:")
	x = ae.relu2.Forward(x)
	fmt.Printf("After relu2: %v\n", x.Shape)

	fmt.Println("\nEncoder FC3 (Latent Space):")
	x = ae.fc3.Forward(x)
	fmt.Printf("After fc3 (latent): %v\n", x.Shape)

	// --- Decoder ---
	fmt.Println("\nDecoder FC4:")
	x = ae.fc4.Forward(x)
	fmt.Printf("After fc4: %v\n", x.Shape)

	fmt.Println("\nReLU3:")
	x = ae.relu3.Forward(x)
	fmt.Printf("After relu3: %v\n", x.Shape)

	fmt.Println("\nDecoder FC5:")
	x = ae.fc5.Forward(x)
	fmt.Printf("After fc5: %v\n", x.Shape)

	fmt.Println("\nReLU4:")
	x = ae.relu4.Forward(x)
	fmt.Printf("After relu4: %v\n", x.Shape)

	fmt.Println("\nDecoder FC6:")
	x = ae.fc6.Forward(x)
	fmt.Printf("After fc6: %v\n", x.Shape)

	fmt.Println("\nSigmoid (Output):")
	x = ae.Sigmoid1.Forward(x)
	fmt.Printf("After sigmoid (output): %v\n", x.Shape) // Should be [N, 784]

	fmt.Println("\n=== AutoEncoder Forward Pass Complete ===")
	return x
}

// Parameters returns a slice of all trainable parameters (weights and biases) in the model.
// Note: Returning *tensor.Tensor based on LinearLayer structure.
func (ae *AutoEncoder) Parameters() []*tensor.Tensor {
	params := make([]*tensor.Tensor, 0)
	// Encoder parameters
	params = append(params, ae.fc1.Weights, ae.fc1.Bias)
	params = append(params, ae.fc2.Weights, ae.fc2.Bias)
	params = append(params, ae.fc3.Weights, ae.fc3.Bias)
	// Decoder parameters
	params = append(params, ae.fc4.Weights, ae.fc4.Bias)
	params = append(params, ae.fc5.Weights, ae.fc5.Bias)
	params = append(params, ae.fc6.Weights, ae.fc6.Bias)
	return params
}

// ZeroGrad resets the gradients of all trainable parameters in the model.
func (ae *AutoEncoder) ZeroGrad() {
	ae.fc1.ZeroGrad()
	ae.fc2.ZeroGrad()
	ae.fc3.ZeroGrad()
	ae.fc4.ZeroGrad()
	ae.fc5.ZeroGrad()
	ae.fc6.ZeroGrad()
	// ReLU and Sigmoid layers typically don't have trainable parameters or gradients to zero.
}

func main() {
	{
		{
			//gen test data
			d, _ := os.Getwd()
			runCommand(filepath.Join(d, "py"), filepath.Join(d, "py", "generate_go_testdata.py"))
		}
		directory := "./py/mnist_noisy_images" // Adjust the path to where your image CSVs and labels.csv are stored

		// Load data (images and labels)
		images, labels, err := LoadDataFromCSVDir(directory)
		if err != nil {
			log.Fatalf("Error loading data: %v", err)
		}

		// Example: Print the first image matrix and its corresponding label
		if len(images) > 0 && len(labels) > 0 {
			fmt.Printf("First Image Data:\n%v\n", images[0])
			fmt.Printf("First Image Label: %s\n", labels[0])
		}

		//num := 4
		for num := 0; num < len(images); num++ {
			model := NewAutoEncoder()
			prediction := Predict(model, images[num])
			prediction.SaveToCSV(filepath.Join(directory, strings.Replace(labels[num], ".png.csv", ".png.denoise.csv", -1)))
			//fmt.Println("Label:", labels[num], "prediction:", prediction.Data[0][0])
		}
		fmt.Println("label:", labels)
		fileName := []string{}
		{

			for i := 0; i < len(labels); i++ {
				d, _ := os.Getwd()
				name := filepath.Join(d, "py", "mnist_noisy_images", labels[i])
				name = strings.Replace(name, ".csv", "", -1)
				fileName = append(fileName, name)
				fmt.Println()
			}
			if !testing.IsTesting() {
				predictPlot(fileName)
			}
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
	data := make([]float32, rows*cols)

	for i, line := range lines {
		for j, value := range line {
			// Convert string to float32
			val, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
			data[i*cols+j] = float32(val)
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
		if !strings.HasSuffix(file.Name(), "png.csv") {
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

func Predict(model *AutoEncoder, image *tensor.Tensor) *tensor.Tensor {
	// Pass the image through the model's forward pass
	image = image.Reshape([]int{1, 1, 28, 28})
	output := model.Forward(image)

	output.Reshape([]int{28, 28})
	return output
}

func predictPlot(imagePaths []string) error {

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
import os
import numpy as np

def load_numpy_from_csv(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        if not header.startswith("Shape,"):
            raise ValueError("Invalid CSV format: missing shape header")
        
        shape = list(map(int, header.split(",")[1:]))
        data = np.loadtxt(f, delimiter=",")
    
    flattened = data.flatten()
    return (flattened).reshape(*shape)


def load_data(file_path):
    image_paths = []
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            image_paths.append(parts[0])
            predictions.append(parts[1])
    return image_paths, predictions

def predict_plot(data_file):
    image_paths, predictions = load_data(data_file)
    num = len(image_paths)
    fig, axes = plt.subplots(2, num, figsize=(15, 5)) 

    for i, (img_path, denoise_csv_path) in enumerate(zip(image_paths, predictions)):
        img = mpimg.imread(img_path)
        axes[0,i].imshow(img, cmap='gray')
        print(denoise_csv_path)
        print(f"Attempting to load: {denoise_csv_path}")
        image_data = load_numpy_from_csv(denoise_csv_path)
        axes[1,i].imshow(image_data, cmap='gray')
        #plt.imshow(image_data, cmap='gray', vmin=0, vmax=1)
        os.remove(denoise_csv_path)
        axes[0,i].axis('off')
        axes[1,i].axis('off')

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
		_, err := dataFile.WriteString(fmt.Sprintf("%s,%s\n", imagePaths[i], imagePaths[i]+".denoise.csv"))
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
