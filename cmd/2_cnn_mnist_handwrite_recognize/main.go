package main

import (
	"encoding/csv"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/pkg/log"
	"github.com/Jimmy2099/torch/testing"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type CNN struct {
	conv1 *torch.ConvLayer
	conv2 *torch.ConvLayer
	fc1   *torch.LinearLayer
	fc2   *torch.LinearLayer
	relu  *torch.ReLULayer
	pool  *torch.MaxPool2DLayer
}

func (c *CNN) Forward(x *tensor.Tensor) *tensor.Tensor {
	fmt.Println("\n=== Starting Forward Pass ===")
	fmt.Printf("Input shape: %v\n", x.GetShape())

	fmt.Println("\nConv1:")
	x = c.conv1.Forward(x)
	fmt.Printf("After conv1: %v\n", x.GetShape())

	fmt.Println("\nReLU1:")
	x = c.relu.Forward(x)
	fmt.Printf("After relu1: %v\n", x.GetShape())

	fmt.Println("\nPool1:")
	x = c.pool.Forward(x)
	fmt.Printf("After pool1: %v\n", x.GetShape())

	fmt.Println("\nConv2:")
	x = c.conv2.Forward(x)
	fmt.Printf("After conv2: %v\n", x.GetShape())

	fmt.Println("\nReLU2:")
	x = c.relu.Forward(x)
	fmt.Printf("After relu2: %v\n", x.GetShape())

	fmt.Println("\nPool2:")
	x = c.pool.Forward(x)
	fmt.Printf("After pool2: %v\n", x.GetShape())

	fmt.Println("\nFlatten:")
	x = x.Flatten()
	fmt.Printf("After flatten: %v\n", x.GetShape())

	fmt.Println("\nFC1:")
	x = c.fc1.Forward(x)
	fmt.Printf("After fc1: %v\n", x.GetShape())

	fmt.Println("\nReLU3:")
	x = c.relu.Forward(x)
	fmt.Printf("After relu3: %v\n", x.GetShape())

	fmt.Println("\nFC2:")
	x = c.fc2.Forward(x)
	fmt.Printf("After fc2: %v\n", x.GetShape())

	fmt.Println("\n=== Forward Pass Complete ===")
	return x
}

func NewCNN() *CNN {
	rand.Seed(time.Now().UnixNano())
	cnn := &CNN{
		conv1: torch.NewConvLayer(1, 32, 3, 1, 1),
		conv2: torch.NewConvLayer(32, 64, 3, 1, 1),
		fc1:   torch.NewLinearLayer(64*7*7, 128),
		fc2:   torch.NewLinearLayer(128, 10),
		relu:  torch.NewReLULayer(),
		pool:  torch.NewMaxPool2DLayer(2, 2, 0),
	}

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

func (c *CNN) ZeroGrad() {
	c.conv1.ZeroGrad()
	c.fc1.ZeroGrad()
	c.fc2.ZeroGrad()
}

func main() {

	{
		d, _ := os.Getwd()
		runCommand(filepath.Join(d, "py"), filepath.Join(d, "py", "generate_go_testdata.py"))
	}
	directory := "./py/mnist_images"

	images, labels, err := LoadDataFromCSVDir(directory)
	if err != nil {
		log.Fatalf("Error loading data: %v", err)
	}

	if len(images) > 0 && len(labels) > 0 {
		fmt.Printf("First Image Matrix:\n%v\n", images[0])
		fmt.Printf("First Image Label: %s\n", labels[0])
	}

	result := []int{}
	for num := 0; num < len(images); num++ {
		model := NewCNN()
		prediction := Predict(model, images[num])
		fmt.Println("Label:", labels[num], "prediction:", prediction.Data[0])
		result = append(result, int(prediction.Data[0]))
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
		return nil, nil, fmt.Errorf("failed to open label file: %v", err)
	}
	defer labelFile.Close()

	reader := csv.NewReader(labelFile)
	labelRecords, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read label CSV: %v", err)
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
		return nil, nil, fmt.Errorf("failed to read directory: %v", err)
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
			log.Printf("failed to read image file %s: %v", file.Name(), err)
			continue
		}

		label := file.Name()

		tensors = append(tensors, image)
		labels = append(labels, label)
	}

	return tensors, labels, nil
}

func Predict(model *CNN, image *tensor.Tensor) *tensor.Tensor {
	image = image.Reshape([]int{1, 1, 28, 28})
	output := model.Forward(image)

	output = output.Reshape([]int{1, len(output.Data)})
	return output.ArgMax()
}

func predictPlot(imagePaths []string, predictions []int) error {
	if len(imagePaths) != len(predictions) {
		return fmt.Errorf("image paths and predictions length mismatch")
	}

	rand.Seed(time.Now().UnixNano())
	suffix := fmt.Sprintf("%d", rand.Intn(1000000))

	tmpScriptFileName := filepath.Join(os.TempDir(), "predict_plot_"+suffix+".py")
	tmpDataFileName := filepath.Join(os.TempDir(), "image_predictions_"+suffix+".txt")

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
    # If there is only one image, convert axes to a list
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
		_, err := dataFile.WriteString(fmt.Sprintf("%s,%d\n", imagePaths[i], predictions[i]))
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
