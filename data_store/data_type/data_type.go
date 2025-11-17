package data_type

type DataType int

const (
	DataTypeUnDefine DataType = iota
	DataTypeFloat16  DataType = iota
	DataTypeFloat32  DataType = iota
)

type Type interface {
	GetDataType() DataType
	Length() int
}

type TypeFloat16 interface {
	Type
}

type TypeFloat32 interface {
	Type
}

type DataMemoryType interface {
	GetData()
	SetData()
	GetShape()
	SetShape()
	String() string
}

type DataMemoryFloat16Type interface {
	DataMemoryType
}

type DataMemoryFloat32Type interface {
	DataMemoryType
}
