package png

import (
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/Jimmy2099/torch/data_store/tensor"
)

func LoadImageToTensor(path string) (*tensor.Tensor, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	return ImageToTensor(img)
}

func ImageToTensor(img image.Image) (*tensor.Tensor, error) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	size := height * width * 3
	data := make([]float32, size)

	t := tensor.NewTensor(data, []int{height, width, 3})

	idx := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			t.Data[idx] = float32(r >> 8)
			t.Data[idx+1] = float32(g >> 8)
			t.Data[idx+2] = float32(b >> 8)

			idx += 3
		}
	}

	return t, nil
}
