package png

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"image"
	"image/color"
	"image/png"
	"os"
)

func TensorToImage(t *tensor.Tensor) image.Image {
	shape := t.GetShape()
	if len(shape) < 2 {
		return nil
	}

	height := shape[0]
	width := shape[1]
	channels := 1

	if len(shape) == 3 {
		channels = shape[2]
	}

	if channels == 1 {
		img := image.NewGray(image.Rect(0, 0, width, height))
		idx := 0
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				v := clampUint8(t.Data[idx])
				img.SetGray(x, y, color.Gray{Y: v})
				idx++
			}
		}
		return img
	}

	if channels == 3 {
		img := image.NewRGBA(image.Rect(0, 0, width, height))
		idx := 0
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r := clampUint8(t.Data[idx])
				g := clampUint8(t.Data[idx+1])
				b := clampUint8(t.Data[idx+2])

				img.SetRGBA(x, y, color.RGBA{
					R: r,
					G: g,
					B: b,
					A: 255,
				})
				idx += 3
			}
		}
		return img
	}

	return nil
}

func clampUint8(v float32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func WriteTempPNG(img image.Image, path string) (err error) {

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	if err = png.Encode(f, img); err != nil {
		return err
	}
	return nil
}
