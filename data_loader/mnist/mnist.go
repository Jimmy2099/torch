package mnist

import (
	"github.com/Jimmy2099/torch/data_struct/matrix"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"math/rand"
	"time"
)

// MNISTData 结构体保存图像和标签
type MNISTData struct {
	Images *matrix.Matrix // 归一化后的像素值 (60000, 784)
	Labels *matrix.Matrix // One-hot 编码的标签 (60000, 10)
}

// LoadMNIST 读取 MNIST 数据集
func LoadMNIST(imageFile, labelFile string) (*MNISTData, error) {
	loadDataset()
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

	// 将[][]float32转换为Matrix
	imageMatrix := matrix.NewMatrixFromSlice(images)
	labelMatrix := matrix.NewMatrixFromSlice(labels)

	return &MNISTData{Images: imageMatrix, Labels: labelMatrix}, nil
}

// MiniBatch 生成批量数据
func (data *MNISTData) MiniBatch(batchSize int) []MNISTData {
	var batches []MNISTData
	for i := 0; i < data.Images.Rows; i += batchSize {
		end := i + batchSize
		if end > data.Images.Rows {
			end = data.Images.Rows
		}

		// 获取子矩阵
		imageBatch := data.Images.GetRows(i, end)
		labelBatch := data.Labels.GetRows(i, end)

		batches = append(batches, MNISTData{
			Images: imageBatch,
			Labels: labelBatch,
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
	indices := rand.Perm(data.Images.Rows)

	// 打乱图像和标签
	data.Images = data.Images.ReorderRows(indices)
	data.Labels = data.Labels.ReorderRows(indices)
}
