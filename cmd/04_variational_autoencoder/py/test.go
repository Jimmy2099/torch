package main

import (
	"github.com/Jimmy2099/torch/data_struct/tensor"
)

func main() {
	x, _ := tensor.LoadFromCSV("test_ok.csv")
	x.Reshape([]int{1, len(x.Data)})
	x.SaveToCSV("test.csv")
}
