package png

import "testing"

func TestLoadAndSave(t *testing.T) {

	tensorImg, err := LoadImageToTensor("../../../images/02_cnn.png")
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	img := TensorToImage(tensorImg)
	err = WriteTempPNG(img, "test.png")
	if err != nil {
		t.Fatalf("WriteTempPNG failed: %v", err)
	}
}
