package algorithm

import (
	"github.com/chewxy/math32"
	"math/rand"
	"sort"
)

func Product(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	p := 1
	for _, dim := range shape {
		if dim <= 0 {
			panic("shape dimension must be positive")
		}
		p *= dim
	}
	return p
}

func ArgMax(data []float32) int {
	maxIndex := 0
	maxValue := data[0]
	for i, v := range data {
		if v > maxValue {
			maxValue = v
			maxIndex = i
		}
	}
	return maxIndex
}

func Softmax(data []float32) []float32 {
	max1 := data[0]
	for _, v := range data {
		if v > max1 {
			max1 = v
		}
	}

	var sum float32
	exp := make([]float32, len(data))
	for i, v := range data {
		exp[i] = math32.Exp(v - max1)
		sum += exp[i]
	}

	for i := range exp {
		exp[i] /= sum
	}
	return exp
}

func TopK(data []float32, k int) ([]float32, []int) {
	type kv struct {
		Value float32
		Index int
	}

	tmp := make([]kv, len(data))
	for i, v := range data {
		tmp[i] = kv{v, i}
	}

	sort.Slice(tmp, func(i, j int) bool {
		return tmp[i].Value > tmp[j].Value
	})

	values := make([]float32, k)
	indices := make([]int, k)
	for i := 0; i < k; i++ {
		values[i] = tmp[i].Value
		indices[i] = tmp[i].Index
	}
	return values, indices
}

func Multinomial(probs []float32) int {
	r := rand.Float32()
	var sum float32
	for i, p := range probs {
		sum += p
		if r < sum {
			return i
		}
	}
	return len(probs) - 1
}
