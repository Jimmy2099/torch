package main

import (
	"encoding/csv"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_store/tensor"
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

type AutoEncoder struct {
	fc1   *torch.LinearLayer
	relu1 *torch.ReLULayer
	fc2   *torch.LinearLayer
	relu2 *torch.ReLULayer
	fc3   *torch.LinearLayer

	fc4      *torch.LinearLayer
	relu3    *torch.ReLULayer
	fc5      *torch.LinearLayer
	relu4    *torch.ReLULayer
	fc6      *torch.LinearLayer
	Sigmoid1 *torch.SigmoidLayer
}

type layerLoadInfo struct {
	pyTorchName string
	goLayer     *torch.LinearLayer
	weightShape []int
	biasShape   []int
}

func NewAutoEncoder() *AutoEncoder {
	fmt.Println("Initializing AutoEncoder layers...")
	ae := &AutoEncoder{
		fc1:   torch.NewLinearLayer(784, 128),
		relu1: torch.NewReLULayer(),
		fc2:   torch.NewLinearLayer(128, 64),
		relu2: torch.NewReLULayer(),
		fc3:   torch.NewLinearLayer(64, 32),

		fc4:      torch.NewLinearLayer(32, 64),
		relu3:    torch.NewReLULayer(),
		fc5:      torch.NewLinearLayer(64, 128),
		relu4:    torch.NewReLULayer(),
		fc6:      torch.NewLinearLayer(128, 784),
		Sigmoid1: torch.NewSigmoidLayer(),
	}
	fmt.Println("Layers initialized.")

	loadInfos := []layerLoadInfo{
		{"encoder.0", ae.fc1, []int{128, 784}, []int{128}},
		{"encoder.2", ae.fc2, []int{64, 128}, []int{64}},
		{"encoder.4", ae.fc3, []int{32, 64}, []int{32}},
		{"decoder.0", ae.fc4, []int{64, 32}, []int{64}},
		{"decoder.2", ae.fc5, []int{128, 64}, []int{128}},
		{"decoder.4", ae.fc6, []int{784, 128}, []int{784}},
	}

	basePath := "py/data"
	d, err := os.Getwd()
	if err != nil {
		panic(fmt.Sprintf("Error getting working directory: %v", err))
	}
	dataPath := filepath.Join(d, basePath)
	fmt.Printf("Base data path set to: %s\n", dataPath)

	fmt.Println("\nLoading AutoEncoder model parameters...")

	for _, info := range loadInfos {
		fmt.Printf("\n--- Loading parameters for: %s ---\n", info.pyTorchName)

		weightFileName := info.pyTorchName + ".weight.csv"
		weightFilePath := filepath.Join(dataPath, weightFileName)
		fmt.Printf("Attempting to load weights from: %s\n", weightFilePath)

		weightMatrix, err := torch.LoadMatrixFromCSV(weightFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading weight file %s: %v", weightFilePath, err))
		}

		info.goLayer.SetWeights(weightMatrix.Data)

		if info.goLayer.Weights == nil {
			panic(fmt.Sprintf("Weight tensor is nil after SetWeights for %s", info.pyTorchName))
		}
		info.goLayer.Weights.Reshape(info.weightShape)
		fmt.Printf("Loaded weights for %s, shape set to: %v\n", info.pyTorchName, info.goLayer.Weights.Shape)

		biasFileName := info.pyTorchName + ".bias.csv"
		biasFilePath := filepath.Join(dataPath, biasFileName)
		fmt.Printf("Attempting to load biases from: %s\n", biasFilePath)

		biasMatrix, err := torch.LoadMatrixFromCSV(biasFilePath)
		if err != nil {
			panic(fmt.Sprintf("Error loading bias file %s: %v", biasFilePath, err))
		}

		info.goLayer.SetBias(biasMatrix.Data)

		if info.goLayer.Bias == nil {
			panic(fmt.Sprintf("Bias tensor is nil after SetBias for %s", info.pyTorchName))
		}
		info.goLayer.Bias.Reshape(info.biasShape)
		fmt.Printf("Loaded biases for %s, shape set to: %v\n", info.pyTorchName, info.goLayer.Bias.Shape)
	}

	fmt.Println("\n--- AutoEncoder model parameters loaded successfully. ---")

	return ae
}

func (ae *AutoEncoder) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 2 || x.Shape[1] != 784 {
		if x.Size() == 784 {
			fmt.Printf("Input shape %v is not [N, 784], attempting to flatten.\n", x.Shape)
			x = x.Flatten()
			fmt.Printf("Flattened input shape: %v\n", x.Shape)
		} else {
			panic(fmt.Sprintf("Input tensor shape %v is incompatible with AutoEncoder input (expected [N, 784])", x.Shape))
		}
	}

	fmt.Println("\n=== Starting AutoEncoder Forward Pass ===")
	fmt.Printf("Input shape: %v\n", x.Shape)

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
	fmt.Printf("After sigmoid (output): %v\n", x.Shape)

	fmt.Println("\n=== AutoEncoder Forward Pass Complete ===")
	return x
}

func (ae *AutoEncoder) Parameters() []*tensor.Tensor {
	params := make([]*tensor.Tensor, 0)
	params = append(params, ae.fc1.Weights, ae.fc1.Bias)
	params = append(params, ae.fc2.Weights, ae.fc2.Bias)
	params = append(params, ae.fc3.Weights, ae.fc3.Bias)
	params = append(params, ae.fc4.Weights, ae.fc4.Bias)
	params = append(params, ae.fc5.Weights, ae.fc5.Bias)
	params = append(params, ae.fc6.Weights, ae.fc6.Bias)
	return params
}

func (ae *AutoEncoder) ZeroGrad() {
	ae.fc1.ZeroGrad()
	ae.fc2.ZeroGrad()
	ae.fc3.ZeroGrad()
	ae.fc4.ZeroGrad()
	ae.fc5.ZeroGrad()
	ae.fc6.ZeroGrad()
}

func main() {
	{
		{
			d, _ := os.Getwd()
			runCommand(filepath.Join(d, "py"), filepath.Join(d, "py", "generate_go_testdata.py"))
		}
		directory := "./py/mnist_noisy_images"

		images, labels, err := LoadDataFromCSVDir(directory)
		if err != nil {
			log.Fatalf("Error loading data: %v", err)
		}

		if len(images) > 0 && len(labels) > 0 {
			fmt.Printf("First Image Data:\n%v\n", images[0])
			fmt.Printf("First Image Label: %s\n", labels[0])
		}

		for num := 0; num < len(images); num++ {
			model := NewAutoEncoder()
			prediction := Predict(model, images[num])
			prediction.SaveToCSV(filepath.Join(directory, strings.Replace(labels[num], ".png.csv", ".png.denoise.csv", -1)))
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

func ReadCSV(filepath string) (*tensor.Tensor, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	rows := len(lines)
	cols := len(lines[0])
	if rows*cols != 784 {
		return nil, fmt.Errorf("expected 784 values in CSV, got %d", rows*cols)
	}

	data := make([]float32, rows*cols)

	for i, line := range lines {
		for j, value := range line {
			val, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
			data[i*cols+j] = float32(val)
		}
	}

	tensorImage := tensor.NewTensor(data, []int{28, 28})

	return tensorImage, nil
}

func LoadDataFromCSVDir(directory string) ([]*tensor.Tensor, []string, error) {
	var tensors []*tensor.Tensor
	var labels []string

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

	labelMap := make(map[string]string)
	for _, record := range labelRecords {
		if len(record) >= 2 {
			filename := strings.TrimSpace(record[0])
			label := strings.TrimSpace(record[1])
			labelMap[filename] = label
		}
	}

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

		label := file.Name()

		tensors = append(tensors, image)
		labels = append(labels, label)
	}

	return tensors, labels, nil
}

func Predict(model *AutoEncoder, image *tensor.Tensor) *tensor.Tensor {
	image = image.Reshape([]int{1, 1, 28, 28})
	output := model.Forward(image)

	output.Reshape([]int{28, 28})
	return output
}

func predictPlot(imagePaths []string) error {

	rand.Seed(time.Now().UnixNano())
	suffix := fmt.Sprintf("%d", rand.Intn(1000000))

	tmpScriptFileName := filepath.Join(os.TempDir(), "predict_plot_"+suffix+".py")
	tmpDataFileName := filepath.Join(os.TempDir(), "image_predictions_"+suffix+".txt")

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
	defer os.Remove(tmpScriptFileName)

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

	cmd := exec.Command("python", tmpScriptFileName, tmpDataFileName)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("error executing Python script: %v", err)
	}

	return nil
}

func runCommand(workSpace string, fileName string) {

	cmd := exec.Command("python", fileName)
	cmd.Dir = workSpace
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(fmt.Sprintln("Error creating Stdout pipe:", err))
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		panic(fmt.Sprintln("Error creating Stderr pipe:", err))
	}

	if err := cmd.Start(); err != nil {
		panic(fmt.Sprintln("Error starting Python script:", err))
	}

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

	if err := cmd.Wait(); err != nil {
		panic(fmt.Sprintln("Error waiting for Python script to finish:", err))
	}
}
