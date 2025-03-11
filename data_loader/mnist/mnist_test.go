package mnist

import (
	"reflect"
	"testing"
)

// 测试 MNIST 数据加载
func TestLoadMNIST(t *testing.T) {
	train, err := LoadMNIST("./dataset/MNIST/raw/train-images-idx3-ubyte", "./dataset/MNIST/raw/train-labels-idx1-ubyte")
	if err != nil {
		t.Fatalf("加载 MNIST 数据失败: %v", err)
	}

	// 检查是否正确加载数据
	if len(train.Images) == 0 || len(train.Labels) == 0 {
		t.Fatalf("数据集为空")
	}

	// 确保第一张图片有 784 个像素
	if len(train.Images[0]) != 784 {
		t.Fatalf("图像数据尺寸错误，期望 784，实际 %d", len(train.Images[0]))
	}

	// 确保标签为 one-hot 编码
	if len(train.Labels[0]) != 10 {
		t.Fatalf("标签数据尺寸错误，期望 10，实际 %d", len(train.Labels[0]))
	}

	t.Logf("MNIST 数据加载测试通过！")
}

// 测试 MiniBatch 生成
func TestMiniBatch(t *testing.T) {
	train, _ := LoadMNIST("./dataset/MNIST/raw/train-images-idx3-ubyte", "./dataset/MNIST/raw/train-labels-idx1-ubyte")

	batches := train.MiniBatch(64)
	if len(batches) == 0 {
		t.Fatalf("MiniBatch 生成失败")
	}

	if len(batches[0].Images) != 64 {
		t.Fatalf("MiniBatch 大小错误，期望 64，实际 %d", len(batches[0].Images))
	}

	t.Logf("MiniBatch 测试通过！")
}

// 测试数据集打乱功能
func TestShuffle(t *testing.T) {
	train, _ := LoadMNIST("./dataset/MNIST/raw/train-images-idx3-ubyte", "./dataset/MNIST/raw/train-labels-idx1-ubyte")

	// 记录打乱前的第一张图片
	firstImage := make([]float64, len(train.Images[0]))
	copy(firstImage, train.Images[0]) // 复制一份，避免修改原数据

	train.Shuffle() // 进行数据集打乱

	// 重新获取第一张图片，检查是否不同（可能相同但概率很低）
	newFirstImage := train.Images[0]

	if reflect.DeepEqual(firstImage, newFirstImage) {
		t.Fatalf("数据集未正确打乱")
	}

	t.Logf("Shuffle 测试通过！")
}
