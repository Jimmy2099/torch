package screen

import "github.com/Jimmy2099/torch/data_store/tensor"

// DrawTensor [Height, Width] (grey) or [Height, Width, 3] (RGB)
func (s *Screen) DrawTensor(t *tensor.Tensor, startX, startY int) {
	if len(t.GetShape()) < 2 {
		return
	}

	height := t.GetShape()[0]
	width := t.GetShape()[1]
	channels := 1

	if len(t.GetShape()) == 3 {
		channels = t.GetShape()[2]
	}

	for r := 0; r < height; r++ {
		for c := 0; c < width; c++ {
			var red, green, blue uint8

			if channels == 1 {
				idx := r*width + c
				val := uint8(t.Data[idx])
				red, green, blue = val, val, val
			} else if channels == 3 {
				idx := (r*width + c) * 3
				red = uint8(t.Data[idx])
				green = uint8(t.Data[idx+1])
				blue = uint8(t.Data[idx+2])
			}

			s.SetPixel(startX+c, startY+r, red, green, blue)
		}
	}
}
