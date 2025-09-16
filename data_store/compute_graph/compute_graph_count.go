package compute_graph

type ComputationalGraphCount struct {
	NameCount map[string]int
}

func NewComputationalGraphCount() *ComputationalGraphCount {
	return &ComputationalGraphCount{
		NameCount: map[string]int{},
	}
}

func (m *ComputationalGraphCount) GetNameAutoInc(name string) int {
	m.NameCount[name] = m.NameCount[name] + 1
	return m.NameCount[name]
}
