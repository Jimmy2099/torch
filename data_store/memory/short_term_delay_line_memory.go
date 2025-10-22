package memory

type ShortTermMemory struct {
	A chan any
	B chan any
}

func NewShortTermMemory() *ShortTermMemory {
	return &ShortTermMemory{
		A: make(chan any, 1),
		B: make(chan any, 1),
	}
}

func (m *ShortTermMemory) Tick() {
	var dataA, dataB any
	var hasA, hasB bool

	select {
	case dataA = <-m.A:
		hasA = true
	default:
	}
	select {
	case dataB = <-m.B:
		hasB = true
	default:
	}

	if hasA {
		select {
		case m.B <- dataA:
		default:
			m.A <- dataA
		}
	}
	if hasB {
		select {
		case m.A <- dataB:
		default:
			m.B <- dataB
		}
	}
}

func (m *ShortTermMemory) SetVal(data any) {
	select {
	case <-m.A:
	default:
	}

	select {
	case m.A <- data:
	default:
	}
}

func (m *ShortTermMemory) Read() any {
	select {
	case data := <-m.A:
		m.A <- data
		return data
	default:
	}

	select {
	case data := <-m.B:
		m.B <- data
		return data
	default:
	}

	return nil
}
