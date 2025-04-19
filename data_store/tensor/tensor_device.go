package tensor

type DeviceType int

const (
	CPU DeviceType = iota
	GPU
)

type Device struct {
	Type  DeviceType
	Index int
}

func GetDefaultDevice() *Device {
	return &Device{
		Type:  CPU,
		Index: 0,
	}
}
