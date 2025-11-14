package testing

type UniqueCheck struct {
	exist map[any]struct{}
}

func NewUnique(any) *UniqueCheck {
	return &UniqueCheck{
		exist: make(map[any]struct{}),
	}
}

func (m *UniqueCheck) Check(name any) bool {
	if _, ok := m.exist[name]; ok {
		return false
	}
	m.exist[name] = struct{}{}
	return true
}
