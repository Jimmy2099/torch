package tensor

type DeviceType int
type DeviceNum int

const (
	DeviceCPU DeviceType = 1 + iota
	DeviceGPU
)

const (
	DefaultDeviceNum DeviceNum = 1
)

type Device interface {
	GetDeviceType() DeviceType
	GetIndex() DeviceNum
}

var defaultDevice = GetCpuMemoryDevice()

func GetDefaultDevice() Device {
	return defaultDevice
}

func SetDefaultDevice(device Device) {
	defaultDevice = device
}
