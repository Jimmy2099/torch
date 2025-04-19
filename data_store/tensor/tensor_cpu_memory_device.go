package tensor

type CpuMemoryDevice struct {
	Type DeviceType
	Num  DeviceNum
}

func GetCpuMemoryDevice() Device {
	device := &CpuMemoryDevice{
		Type: DeviceCPU,
		Num:  DefaultDeviceNum,
	}
	return device
}

func (d *CpuMemoryDevice) GetDeviceType() DeviceType {
	return d.Type
}

func (d *CpuMemoryDevice) GetIndex() DeviceNum {
	return d.Num
}
