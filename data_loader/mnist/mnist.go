package mnist

import (
	"fmt"
	"math/rand"
	"time"
)

// MNISTData 结构体保存图像和标签
type MNISTData struct {
	Images [][]float64 // 归一化后的像素值 (60000, 784)
	Labels [][]float64 // One-hot 编码的标签 (60000, 10)
}

// LoadMNIST 读取 MNIST 数据集
func LoadMNIST(imageFile, labelFile string) (*MNISTData, error) {
	LoadDataset()
	images, err := loadImages(imageFile)
	if err != nil {
		return nil, err
	}
	if images == nil {
		return nil, fmt.Errorf("images data is nil")
	}

	labels, err := loadLabels(labelFile)
	if err != nil {
		return nil, err
	}
	if labels == nil {
		return nil, fmt.Errorf("labels data is nil")
	}

	return &MNISTData{Images: images, Labels: labels}, nil
}

// MiniBatch 生成批量数据
func (data *MNISTData) MiniBatch(batchSize int) []MNISTData {
	var batches []MNISTData
	for i := 0; i < len(data.Images); i += batchSize {
		end := i + batchSize
		if end > len(data.Images) {
			end = len(data.Images)
		}
		batches = append(batches, MNISTData{
			Images: data.Images[i:end],
			Labels: data.Labels[i:end],
		})
	}
	return batches
}

// Shuffle 打乱数据集
func (data *MNISTData) Shuffle() {
	if data.Images == nil || data.Labels == nil {
		panic("Images or Labels are nil")
	}

	rand.Seed(time.Now().UnixNano())
	for i := range data.Images {
		j := rand.Intn(len(data.Images))
		data.Images[i], data.Images[j] = data.Images[j], data.Images[i]
		data.Labels[i], data.Labels[j] = data.Labels[j], data.Labels[i]
	}
}
