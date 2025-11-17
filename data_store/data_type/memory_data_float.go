package data_type

type DataMemoryFloat struct {
	DataMemory
}

func NewDataMemoryFloat32() DataMemoryFloat32Type {
	return &DataMemoryFloat{}
}
