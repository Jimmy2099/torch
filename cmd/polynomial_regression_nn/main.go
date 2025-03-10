package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/matrix"
)

// PolyFitModel 实现多项式拟合模型
type PolyFitModel struct {
	a, b, c float64 // 多项式系数
}

func NewPolyFitModel() *PolyFitModel {
	// 使用更合理的初始化范围
	return &PolyFitModel{
		a: rand.NormFloat64() * 0.1,
		b: rand.NormFloat64() * 0.1,
		c: rand.NormFloat64() * 0.1,
	}
}

func (m *PolyFitModel) Forward(input *matrix.Matrix) *matrix.Matrix {
	// 计算 y = a*x^2 + b*x + c
	result := matrix.NewMatrix(input.Rows, 1)
	for i := 0; i < input.Rows; i++ {
		x := input.Data[i][0]
		result.Data[i][0] = m.a*math.Pow(x, 2) + m.b*x + m.c
	}
	return result
}

func (m *PolyFitModel) Backward(target *matrix.Matrix, learningRate float64) {
	// 计算梯度
	pred := m.Forward(target)
	diff := matrix.Subtract(pred, target)
	
	// 计算每个参数的梯度
	gradA := 0.0
	gradB := 0.0
	gradC := 0.0
	
	for i := 0; i < target.Rows; i++ {
		x := target.Data[i][0]
		gradA += diff.Data[i][0] * math.Pow(x, 2)
		gradB += diff.Data[i][0] * x
		gradC += diff.Data[i][0]
	}
	
	// 平均梯度
	gradA /= float64(target.Rows)
	gradB /= float64(target.Rows)
	gradC /= float64(target.Rows)
	
	// 更新参数
	m.a -= learningRate * gradA
	m.b -= learningRate * gradB
	m.c -= learningRate * gradC
}

func (m *PolyFitModel) Parameters() []*matrix.Matrix {
	// 返回空切片，因为我们直接存储参数
	return []*matrix.Matrix{}
}

func (m *PolyFitModel) ZeroGrad() {
	// 无需实现，因为我们直接存储参数
}

func main() {
	// 生成数据
	rand.Seed(42)
	xData := linspace(-10, 10, 100)
	yData := make([]float64, len(xData))
	for i, x := range xData {
		yData[i] = 3*math.Pow(x, 2) - 2*x + 5 + rand.NormFloat64()*5
	}

	// 转换为矩阵
	xMatrix := matrix.NewMatrix(len(xData), 1)
	for i, x := range xData {
		xMatrix.Data[i][0] = x
	}
	
	yMatrix := matrix.NewMatrix(len(yData), 1)
	for i, y := range yData {
		yMatrix.Data[i][0] = y
	}

	// 创建模型
	model := NewPolyFitModel()
	trainer := torch.NewBasicTrainer(torch.MSE)

	// 训练模型
	epochs := 5000
	learningRate := 0.0001  // 适当提高学习率
	trainer.Train(model, xMatrix, yMatrix, epochs, learningRate)

	// 预测并打印部分结果
	pred := model.Forward(xMatrix)
	fmt.Println("\nPredictions vs Actual:")
	for i := 0; i < 5; i++ { // 只打印前5个样本
		fmt.Printf("x: %.2f, Pred: %.2f, Actual: %.2f\n",
			xData[i], pred.Data[i][0], yData[i])
	}

	// 计算并打印最终损失
	finalLoss := torch.MSE(pred, yMatrix)
	fmt.Printf("\nFinal Loss: %.4f\n", finalLoss)

	// 输出拟合参数
	fmt.Printf("\n拟合的参数: a = %.4f, b = %.4f, c = %.4f\n", model.a, model.b, model.c)
}

// linspace 生成等间距数组
func linspace(start, end float64, num int) []float64 {
	result := make([]float64, num)
	step := (end - start) / float64(num-1)
	for i := range result {
		result[i] = start + float64(i)*step
	}
	return result
}
