package algorithm

func Relu(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

func ReluDerivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}
