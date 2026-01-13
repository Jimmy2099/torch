package main

import (
	"encoding/csv"
	"github.com/Jimmy2099/torch/data_store/compute_graph"
	"github.com/Jimmy2099/torch/data_store/network"
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
	graph *compute_graph.ComputationalGraph
}

func transpose2D(data []float32, rows, cols int) ([]float32, []int) {
	transposed := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j*rows+i] = data[i*cols+j]
		}
	}
	return transposed, []int{cols, rows}
}

func loadParameters() (map[string]*tensor.Tensor, map[string]*tensor.Tensor) {
	weights := make(map[string]*tensor.Tensor)
	biases := make(map[string]*tensor.Tensor)

	layerSpecs := []struct {
		name        string
		weightShape []int
		biasShape   []int
	}{
		{"encoder.0", []int{128, 784}, []int{128}},
		{"encoder.2", []int{64, 128}, []int{64}},
		{"encoder.4", []int{32, 64}, []int{32}},
		{"decoder.0", []int{64, 32}, []int{64}},
		{"decoder.2", []int{128, 64}, []int{128}},
		{"decoder.4", []int{784, 128}, []int{784}},
	}

	basePath, err := filepath.Abs("./py/data")
	if err != nil {
		panic(err)
	}

	fmt.Printf("Loading parameters from: %s\n", basePath)

	for _, spec := range layerSpecs {
		weightPath := filepath.Join(basePath, spec.name+".weight.csv")
		weightTensor, err := loadTensorFromCSV(weightPath, spec.weightShape)
		if err != nil {
			panic(fmt.Sprintf("Load weight %s failed: %v", weightPath, err))
		}
		transposedData, transposedShape := transpose2D(weightTensor.Data, spec.weightShape[0], spec.weightShape[1])
		weights[spec.name] = tensor.NewTensor(transposedData, transposedShape)

		biasPath := filepath.Join(basePath, spec.name+".bias.csv")
		biasTensor, err := loadTensorFromCSV(biasPath, spec.biasShape)
		if err != nil {
			panic(fmt.Sprintf("Load bias %s failed: %v", biasPath, err))
		}
		biases[spec.name] = biasTensor
	}
	return weights, biases
}

func NewAutoEncoder() *AutoEncoder {
	fmt.Println("Initializing AutoEncoder with computational graph...")
	graph := compute_graph.NewComputationalGraph()
	net := graph.Network
	nodes := make(map[string]*network.Node)

	createNode := func(name, nodeType string) *network.Node {
		node := net.NewNode()
		node.Name = name
		node.Type = nodeType
		nodes[name] = node
		return node
	}

	createNode("input", "Tensor_Input")
	paramNames := []string{"encoder.0", "encoder.2", "encoder.4", "decoder.0", "decoder.2", "decoder.4"}
	for _, p := range paramNames {
		createNode(p+".weight", "Tensor_Weight")
		createNode(p+".bias", "Tensor_Bias")
	}

	fmt.Println("Building computation graph blueprint...")
	currentX := "input"

	buildLayer := func(inputName, layerName, activation string) string {
		matmulNode := createNode(layerName+"_matmul", "MatMul")
		matmulOut := createNode(layerName+"_matmul_out", "Tensor_Hidden")
		matmulNode.AddInput(nodes[inputName])
		matmulNode.AddInput(nodes[layerName+".weight"])
		matmulNode.AddOutput(matmulOut)

		addNode := createNode(layerName+"_add", "Add")
		addOut := createNode(layerName+"_add_out", "Tensor_Hidden")
		addNode.AddInput(matmulOut)
		addNode.AddInput(nodes[layerName+".bias"])
		addNode.AddOutput(addOut)

		if activation != "" {
			actNode := createNode(layerName+"_"+activation, activation)
			actOut := createNode(layerName+"_"+activation+"_out", "Tensor_Hidden")
			actNode.AddInput(addOut)
			actNode.AddOutput(actOut)
			return actOut.Name
		}
		return addOut.Name
	}

	currentX = buildLayer(currentX, "encoder.0", "Relu")
	currentX = buildLayer(currentX, "encoder.2", "Relu")
	currentX = buildLayer(currentX, "encoder.4", "")

	currentX = buildLayer(currentX, "decoder.0", "Relu")
	currentX = buildLayer(currentX, "decoder.2", "Relu")
	currentX = buildLayer(currentX, "decoder.4", "Sigmoid")

	net.AddInput(nodes["input"])
	for _, p := range paramNames {
		net.AddInput(nodes[p+".weight"])
		net.AddInput(nodes[p+".bias"])
	}
	net.AddOutput(nodes[currentX])

	fmt.Println("\nLoading and setting parameters...")
	weights, biases := loadParameters()
	for name, tensorData := range weights {
		graph.GetTensorByName(name + ".weight").SetValue(tensorData)
	}
	for name, tensorData := range biases {
		graph.GetTensorByName(name + ".bias").SetValue(tensorData)
	}

	ae := &AutoEncoder{
		graph: graph,
	}

	fmt.Println("Computation graph structure:")
	return ae
}

func (ae *AutoEncoder) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x.Size() != 784 {
		panic("Invalid input size, must be 784")
	}
	x.Reshape([]int{1, 784})

	ae.graph.GetTensorByName("input").SetValue(x)

	ae.graph.Forward()

	outputTensor := ae.graph.GetTensorByName(ae.graph.Network.GetOutputName()[0])
	return outputTensor.Value()
}

func loadTensorFromCSV(path string, shape []int) (*tensor.Tensor, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	data := make([]float32, 0, len(records)*len(records[0]))
	for _, row := range records {
		for _, val := range row {
			f, err := strconv.ParseFloat(val, 32)
			if err != nil {
				return nil, err
			}
			data = append(data, float32(f))
		}
	}

	return tensor.NewTensor(data, shape), nil
}

func main() {
	{
		directory := "./py/mnist_noisy_images"

		if _, err := os.Stat(directory); err != nil {
			d, _ := os.Getwd()
			runCommand(filepath.Join(d, "py"), filepath.Join(d, "py", "generate_go_testdata.py"))
		}

		images, labels, err := LoadDataFromCSVDir(directory)
		if err != nil {
			log.Fatalf("Error loading data: %v", err)
		}

		if len(images) > 0 && len(labels) > 0 {
			fmt.Printf("First Image Data:\n%v\n", images[0])
			fmt.Printf("First Image Label: %s\n", labels[0])
		}

		model := NewAutoEncoder()

		{
			model.graph.ComputeDependencyGraph.ComputeSortedNodes()
			err = model.graph.Validate()
			if err != nil {
				panic(err)
			}
		}

		for num := 0; num < len(images); num++ {
			prediction := Predict(model, images[num])
			prediction.SaveToCSV(filepath.Join(directory, strings.Replace(labels[num], ".png.csv", ".png.denoise.csv", -1)))
		}

		{
			var onnx *compute_graph.ONNX
			onnx, err = model.graph.ToONNXModel()
			if err != nil {
				log.Fatalf("Error creating ONNX model: %v", err)
			}
			err = onnx.SaveONNX("model.onnx")
			if err != nil {
				log.Fatalf("Error saving ONNX model: %v", err)
			}
		}

		fmt.Println("label:", labels)
		var fileName []string
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
		if !strings.HasSuffix(file.Name(), "png.csv") {
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
        if not header.startswith("shape,"):
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
		_, err = dataFile.WriteString(fmt.Sprintf("%s,%s\n", imagePaths[i], imagePaths[i]+".denoise.csv"))
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

	if err = cmd.Run(); err != nil {
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

	if err = cmd.Start(); err != nil {
		panic(fmt.Sprintln("Error starting Python script:", err))
	}

	go func() {
		_, err = io.Copy(os.Stdout, stdout)
		if err != nil {
			panic(fmt.Sprintln("Error copying stdout:", err))
		}
	}()

	go func() {
		_, err = io.Copy(os.Stderr, stderr)
		if err != nil {
			panic(fmt.Sprintln("Error copying stderr:", err))
		}
	}()

	if err = cmd.Wait(); err != nil {
		panic(fmt.Sprintln("Error waiting for Python script to finish:", err))
	}
}
