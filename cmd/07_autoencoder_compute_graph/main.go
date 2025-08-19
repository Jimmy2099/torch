package main

import (
	"encoding/csv"
	"github.com/Jimmy2099/torch/data_store/compute_graph"
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
	graph      *compute_graph.ComputationalGraph
	inputNode  *compute_graph.GraphTensor
	outputNode *compute_graph.GraphTensor
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

func loadParameters(graph *compute_graph.ComputationalGraph) (map[string]*compute_graph.GraphTensor, map[string]*compute_graph.GraphTensor) {
	weightNodes := make(map[string]*compute_graph.GraphTensor)
	biasNodes := make(map[string]*compute_graph.GraphTensor)

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

	basePath := ""
	if d, err := os.Getwd(); err == nil {
		basePath = filepath.Join(d, "py/data")
	} else {
		panic(err)
	}

	fmt.Printf("Loading parameters from: %s\n", basePath)

	for _, spec := range layerSpecs {
		weightPath := filepath.Join(basePath, spec.name+".weight.csv")
		fmt.Printf("Loading weight: %s\n", weightPath)
		weightTensor, err := loadTensorFromCSV(weightPath, spec.weightShape)
		if err != nil {
			panic(fmt.Sprintf("Load weight %s failed: %v", weightPath, err))
		}

		fmt.Printf("Loaded weight %s: shape %v, data[0:5] = %v\n",
			spec.name, weightTensor.GetShape(), weightTensor.Data[:5])

		transposedData, transposedShape := transpose2D(
			weightTensor.Data,
			spec.weightShape[0],
			spec.weightShape[1],
		)
		fmt.Printf("Transposed weight %s: new shape %v\n", spec.name, transposedShape)

		weightNodes[spec.name] = graph.NewGraphTensor(
			transposedData,
			transposedShape,
			spec.name+".weight_transposed",
		)

		biasPath := filepath.Join(basePath, spec.name+".bias.csv")
		fmt.Printf("Loading bias: %s\n", biasPath)
		biasTensor, err := loadTensorFromCSV(biasPath, spec.biasShape)
		if err != nil {
			panic(fmt.Sprintf("Load bias %s failed: %v", biasPath, err))
		}

		fmt.Printf("Loaded bias %s: shape %v, data[0:5] = %v\n",
			spec.name, biasTensor.GetShape(), biasTensor.Data[:5])

		biasNodes[spec.name] = graph.NewGraphTensor(
			biasTensor.Data,
			spec.biasShape,
			spec.name+".bias",
		)
	}
	return weightNodes, biasNodes
}

func NewAutoEncoder() *AutoEncoder {
	fmt.Println("Initializing AutoEncoder with computational graph...")
	ae := &AutoEncoder{
		graph: compute_graph.NewComputationalGraph(),
	}

	inputData := make([]float32, 784)
	ae.inputNode = ae.graph.NewGraphTensor(inputData, []int{1, 784}, "input")
	fmt.Printf("Input node shape: %v\n", ae.inputNode.Value().GetShape())

	weightNodes, biasNodes := loadParameters(ae.graph)

	x := ae.inputNode
	fmt.Println("\nBuilding computation graph...")

	fmt.Printf("First layer weight shape: %v\n", weightNodes["encoder.0"].Value().GetShape())

	buildLinear := func(x *compute_graph.GraphTensor, weight, bias *compute_graph.GraphTensor, name string) *compute_graph.GraphTensor {
		fmt.Printf("\nBuilding layer: %s", name)
		fmt.Printf("\n  Input shape: %v", x.Shape)
		fmt.Printf("\n  Weight shape: %v", weight.Shape)

		//return x
		matmul := x.MatMul(weight, name+"_matmul")
		fmt.Printf("\n  After matmul: %v", matmul.Shape)

		add := matmul.Add(bias, name+"_add")
		fmt.Printf("\n  After add: %v", add.Shape)

		return add
	}

	fmt.Println("\n\nBuilding encoder...")
	x = buildLinear(x, weightNodes["encoder.0"], biasNodes["encoder.0"], "encoder_fc1")
	x = x.ReLU("encoder_relu1")
	fmt.Printf("\nAfter ReLU1: %v", x.Value().GetShape())

	x = buildLinear(x, weightNodes["encoder.2"], biasNodes["encoder.2"], "encoder_fc2")
	x = x.ReLU("encoder_relu2")
	fmt.Printf("\nAfter ReLU2: %v", x.Value().GetShape())

	x = buildLinear(x, weightNodes["encoder.4"], biasNodes["encoder.4"], "encoder_fc3")
	fmt.Printf("\nLatent space: %v", x.Value().GetShape())

	fmt.Println("\n\nBuilding decoder...")
	x = buildLinear(x, weightNodes["decoder.0"], biasNodes["decoder.0"], "decoder_fc4")
	x = x.ReLU("decoder_relu3")
	fmt.Printf("\nAfter ReLU3: %v", x.Value().GetShape())

	x = buildLinear(x, weightNodes["decoder.2"], biasNodes["decoder.2"], "decoder_fc5")
	x = x.ReLU("decoder_relu4")
	fmt.Printf("\nAfter ReLU4: %v", x.Value().GetShape())

	x = buildLinear(x, weightNodes["decoder.4"], biasNodes["decoder.4"], "decoder_fc6")
	x = x.Sigmoid("output_sigmoid")
	fmt.Printf("\nOutput shape: %v", x.Value().GetShape())

	ae.outputNode = x
	ae.graph.SetOutput(x)

	onnxModel, err := ae.graph.ToONNXModel()
	if err != nil {
		panic(err)
	}

	onnxModel.SaveONNX("ae_model.onnx")

	fmt.Println("\n\nComputation graph structure:")
	ae.graph.PrintStructure()

	return ae
}

func (ae *AutoEncoder) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.GetShape()) != 2 || x.GetShape()[1] != 784 {
		if x.Size() == 784 {
			x = x.Flatten().Reshape([]int{1, 784})
		} else {
			panic("Invalid input shape")
		}
	}

	ae.graph.SetInput(ae.inputNode, x.Data)

	ae.graph.Forward()
	return ae.graph.GetOutput().Value()
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
