package tensor

type CpuMemoryDevice struct {
	Type DeviceType
	Num  int
}

func GetCpuMemoryDevice() Device {
	device := &CpuMemoryDevice{
		Type: CPU,
		Num:  0,
	}
	return device
}

func (d *CpuMemoryDevice) GetDeviceType() DeviceType {
	return d.Type
}

func (d *CpuMemoryDevice) GetIndex() int {
	return d.Num
}
