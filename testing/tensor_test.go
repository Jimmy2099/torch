package testing

import (
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"testing"
)

func TestGetTensorTestResult(t *testing.T) {
	// 创建测试输入张量
	inTensor := tensor.Ones([]int{2, 2})

	// 修正后的 Python 计算代码（改为乘法）
	pythonScript := `out = in1 * in2`

	// 调用 GetTensorTestResult
	resultTensor := GetTensorTestResult(pythonScript, inTensor, inTensor)

	// 期望输出（保持原逻辑）
	expectedTensor := inTensor.Mul(inTensor)

	// 添加断言验证结果
	if !resultTensor.Equal(expectedTensor) {
		t.Errorf("Test failed:\nExpected:\n%v\nGot:\n%v", expectedTensor, resultTensor)
	}
}
