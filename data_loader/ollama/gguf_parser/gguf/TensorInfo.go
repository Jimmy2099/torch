package gguf

import (
	"io"
)

type TensorInfo struct {
	g *Reader

	Name string

	Dimensions []uint64

	Type GGML

	Offset uint64
}

func (t *TensorInfo) Reader() (io.Reader, error) {
	_, err := t.g.r.Seek(t.g.tensorOffset, io.SeekStart)
	if err != nil {
		return nil, err
	}

	_, err = t.g.r.Seek(int64(t.Offset), io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	return t.g.r, nil
}

func (t *TensorInfo) Size() int64 {
	s, found := sizes[t.Type]
	if !found {
		panic("unknown type: " + t.Type.String())
	}

	values := uint64(1)

	for _, d := range t.Dimensions {
		values *= d
	}

	return int64((values / s.valuesinblock) * s.blocksize)
}
