package main

import (
	"encoding/csv"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"log"
	"os"
	"strconv"
)

// 读取 CSV 文件并返回浮动数值的切片
func readCSV(filename string) ([]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var data []float32

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

func main() {
	// 读取权重和偏置的 CSV 文件
	weightData, err := readCSV("fc1_weight.csv")
	if err != nil {
		log.Fatal(err)
	}
	biasData, err := readCSV("fc1_bias.csv")
	if err != nil {
		log.Fatal(err)
	}

	// 创建矩阵（假设 fc1 权重是 128x784，偏置是 128）
	weights := mat64.NewDense(128, 784, weightData)
	biases := mat64.NewDense(1, 128, biasData)

	// 获取矩阵的维度
	rows, cols := weights.Dims()
	fmt.Printf("weights维度: %d行, %d列\n", rows, cols)
	biasRows, biasCols := biases.Dims()
	fmt.Printf("biases维度: %d行, %d列\n", biasRows, biasCols)

	// 创建一个输入向量（784个元素），此处用全零初始化
	inputData := make([]float32, 784)
	input := mat64.NewDense(1, 784, inputData)

	// 计算线性变换：output = input * weights^T + biases
	var output mat64.Dense
	output.Mul(input, weights.T()) // input: 1x784, weights.T(): 784x128, 结果为1x128
	output.Add(&output, biases)    // 与偏置（1x128）相加

	// 打印输出
	fmt.Printf("输出结果: %v\n", output)
}
