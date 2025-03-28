package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/abrander/gguf"
)

// 函数：打印标量类型的元数据
func print[T any](name string, val T) {
	fmt.Printf("Metadata: %s: \033[33m%v\033[0m\n", name, val)
}

// 函数：打印数组类型的元数据
func printArray[T any](name string, val []T) {
	fmt.Printf("Metadata: %s: [\033[32m%d\033[0m]\033[36m%T\033[0m\n", name, len(val), val[0])
}

// 函数：计算总参数量
func calculateTotalParams(tensors []gguf.TensorInfo) uint64 {
	totalParams := uint64(0) // 使用 uint64 类型
	for _, t := range tensors {
		params := uint64(1) // 将 params 定义为 uint64 类型
		for _, dim := range t.Dimensions {
			params *= dim // 乘法操作不再会报错
		}
		totalParams += params
	}
	return totalParams
}

func main() {
	if len(os.Args) != 2 {
		fmt.Printf("Usage: %s <file>\n", os.Args[0])
		os.Exit(1)
	}

	// 打开GGUF文件
	g, err := gguf.OpenFile(os.Args[1])
	if err != nil {
		panic(err)
	}

	// 获取所有元数据的键
	keys := make([]string, 0, len(g.Metadata))
	for k := range g.Metadata {
		keys = append(keys, k)
	}

	// 打印文件的一些基本信息
	fmt.Printf("File byte order: \033[33m%v\033[0m\n", g.ByteOrder.String())
	fmt.Printf("File Version: \033[33m%d\033[0m\n", g.Version)

	// 排序元数据的键并打印
	sort.StringSlice(keys).Sort()

	for _, k := range keys {
		v := g.Metadata[k]

		switch vv := v.(type) {
		// 打印标量类型
		case uint8, int8, uint16, int16, uint32, int32, float32, bool, string, uint64, int64, float64, fmt.Stringer:
			print(k, vv)

		// 打印数组类型
		case []uint8:
			printArray(k, vv)
		case []int8:
			printArray(k, vv)
		case []uint16:
			printArray(k, vv)
		case []int16:
			printArray(k, vv)
		case []uint32:
			printArray(k, vv)
		case []int32:
			printArray(k, vv)
		case []float32:
			printArray(k, vv)
		case []bool:
			printArray(k, vv)
		case []string:
			printArray(k, vv)
		case []uint64:
			printArray(k, vv)
		case []int64:
			printArray(k, vv)
		case []float64:
			printArray(k, vv)

		default:
			fmt.Printf("%s: %T\n", k, v)
		}
	}

	// 打印所有张量信息
	for _, t := range g.Tensors {
		dims := make([]string, len(t.Dimensions))

		for i, d := range t.Dimensions {
			dims[i] = fmt.Sprintf("\033[32m%d\033[0m", d)
		}

		fmt.Printf("Tensor: %s: \033[36m%s\033[0m [%s]\n", t.Name, t.Type, strings.Join(dims, "×"))
	}

	// 计算并打印总参数量
	totalParams := calculateTotalParams(g.Tensors)
	fmt.Printf("\033[33mTotal model parameters: %d\033[0m\n", totalParams)
}
