package testing

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"testing"
)

func TestGetTensorTestResult(t *testing.T) {
	// 创建测试输入张量
	inTensor := tensor.Ones([]int{2, 2})

	// 计算平方的 Python 代码
	pythonScript := `
	out= in1 + in2
	`

	// 调用 GetTensorTestResult
	resultTensor := GetTensorTestResult(pythonScript, inTensor, inTensor)

	// 期望输出
	outTensor := tensor.Ones([]int{2, 2})
	outTensor = inTensor.Mul(inTensor)

	fmt.Println(resultTensor, outTensor)

}
