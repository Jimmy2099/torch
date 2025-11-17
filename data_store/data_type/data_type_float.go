package data_type

type Number struct {
	sign        int //bit
	exponent    int //bit
	faction     int //bit
	dataType    DataType
	lengthCache int
}

func NewNumberFloat16() TypeFloat16 {
	return &Number{
		sign:     1,
		exponent: 5,
		faction:  10,
		dataType: DataTypeFloat16,
	}
}

func NewNumberFloat32() TypeFloat32 {
	return &Number{
		sign:     1,
		exponent: 7,
		faction:  23,
		dataType: DataTypeFloat32,
	}
}

func (m *Number) GetDataType() DataType {
	return m.dataType
}

func (m *Number) Length() int {
	if m.lengthCache != 0 {
		return m.lengthCache
	}

	m.lengthCache = m.sign + m.exponent + m.faction

	return m.lengthCache
}
