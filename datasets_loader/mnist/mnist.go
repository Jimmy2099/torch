package mnist

import (
	"fmt"
	//"github.com/Jimmy2099/torch/data_store/matrix"
	//"github.com/Jimmy2099/torch/pkg/fmt"
	"math/rand"
	"time"
)

type MNISTData struct {
	Images *matrix.Matrix
	Labels *matrix.Matrix
}

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

	imageMatrix := matrix.NewMatrixFromSlice(images)
	labelMatrix := matrix.NewMatrixFromSlice(labels)

	return &MNISTData{Images: imageMatrix, Labels: labelMatrix}, nil
}

func (data *MNISTData) MiniBatch(batchSize int) []MNISTData {
	var batches []MNISTData
	for i := 0; i < data.Images.Rows; i += batchSize {
		end := i + batchSize
		if end > data.Images.Rows {
			end = data.Images.Rows
		}

		imageBatch := data.Images.GetRows(i, end)
		labelBatch := data.Labels.GetRows(i, end)

		batches = append(batches, MNISTData{
			Images: imageBatch,
			Labels: labelBatch,
		})
	}
	return batches
}

func (data *MNISTData) Shuffle() {
	if data.Images == nil || data.Labels == nil {
		panic("Images or Labels are nil")
	}

	rand.Seed(time.Now().UnixNano())
	indices := rand.Perm(data.Images.Rows)

	data.Images = data.Images.ReorderRows(indices)
	data.Labels = data.Labels.ReorderRows(indices)
}
