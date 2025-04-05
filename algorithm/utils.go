package algorithm

// Product 计算形状各维度的乘积（总元素数）
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
