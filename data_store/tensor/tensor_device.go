package tensor

type DeviceType int

const (
	CPU DeviceType = iota
	GPU
)

type Device interface {
	GetDeviceType() DeviceType
	GetIndex() int
}

var defaultDevice = GetCpuMemoryDevice()

func GetDefaultDevice() Device {
	return defaultDevice
}

func SetDefaultDevice(device Device) {
	defaultDevice = device
}
