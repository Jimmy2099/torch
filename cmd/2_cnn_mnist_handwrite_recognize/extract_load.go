package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

// 读取 CSV 文件并返回浮动数值的切片
func readCSV(filename string) ([]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var data []float64

	// 读取每一行
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		for _, value := range record {
			val, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
			data = append(data, val)
		}
	}
	return data, nil
}

//func loadData(filename string) (weightData, biasData []float64, err error) {
//	d, _ := os.Getwd()
//	weightData, err = readCSV(filepath.Join(d, "py", "data", filename+".weight.csv"))
//	if err != nil {
//		log.Fatal(err)
//	}
//	biasData, err = readCSV(filepath.Join(d, "py", "data", filename+".bias.csv"))
//	if err != nil {
//		log.Fatal(err)
//	}
//	return weightData, biasData, nil
//}

func (m *CNN) LoadModelFromCSV() {
	layer := []string{
		"conv1",
		"conv2",
		"fc1",
		"fc2",
	}
	d, _ := os.Getwd()

	for k, filename := range layer {
		fmt.Println("loading", k, filename)

		//{
		//	d, _ := os.Getwd()
		//	weightData, err := torch.LoadMatrixFromCSV(filepath.Join(d, "py", "data", filename+".weight.csv"))
		//	if err != nil {
		//		panic(err)
		//	}
		//	biasData, err := torch.LoadMatrixFromCSV(filepath.Join(d, "py", "data", filename+".bias.csv"))
		//	if err != nil {
		//		panic(err)
		//	}
		//}

		fmt.Println("loaded", k, filename)

	}
	fmt.Println(d)
	//        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
	//        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
	//        self.fc1 = torch.nn.Linear(in_features=64 * 7 * 7, out_features=128)
	//        self.fc2 = torch.nn.Linear(in_features=128, out_features=10)
	//        self.relu = torch.nn.ReLU()
	//        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

	//torch.NewConvLayer(1, 32, 3, 3, 1)
	//torch.NewConvLayer(32, 64, 3, 1, 1)
	//torch.NewLinearLayer(64*7*7, 128)
	//torch.NewLinearLayer(128, 10)
	//torch.NewReLULayer()
	//torch.NewMaxPool2DLayer(2, 2, 0)
}
