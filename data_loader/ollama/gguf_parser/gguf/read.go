package gguf

import (
	"encoding/binary"
	"io"
)

type readables interface {
	~uint8 | ~int8 | ~uint16 | ~int16 | ~uint32 | ~int32 | ~uint64 | ~int64 | ~float32 | ~float32
}

func read[T readables](r io.Reader, byteorder binary.ByteOrder) (T, error) {
	var v T

	err := binary.Read(r, byteorder, &v)

	return v, err
}

func readCast[T, C readables](r io.Reader, byteorder binary.ByteOrder) (C, error) {
	var v T

	err := binary.Read(r, byteorder, &v)

	return C(v), err
}
