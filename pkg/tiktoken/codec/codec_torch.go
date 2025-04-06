package codec

// This file is added by the Torch library author, and the license is also under the Torch library license.

import (
	"github.com/dlclark/regexp2"
)

func NewCodec(name string, vocabulary vocab, specialTokens map[string]uint, splitRegexp *regexp2.Regexp) *Codec {
	return &Codec{
		name:          name,
		vocabulary:    vocabulary,
		splitRegexp:   splitRegexp,
		specialTokens: specialTokens,
	}
}
