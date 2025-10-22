package memory

import (
	"testing"
	"time"
)

func TestNewShortTermMemory(t *testing.T) {
	m := NewShortTermMemory()

	if m == nil {
		t.Fatal("NewShortTermMemory() returned nil")
	}
	if m.A == nil {
		t.Error("Channel A was not initialized")
	}
	if m.B == nil {
		t.Error("Channel B was not initialized")
	}
	if cap(m.A) != 1 {
		t.Errorf("Expected capacity of channel A to be 1, but got %d", cap(m.A))
	}
	if cap(m.B) != 1 {
		t.Errorf("Expected capacity of channel B to be 1, but got %d", cap(m.B))
	}
}

func TestSetAndRead(t *testing.T) {
	m := NewShortTermMemory()

	if val := m.Read(); val != nil {
		t.Errorf("Expected to read nil from empty memory, but got %v", val)
	}

	testValue := "hello"
	m.SetVal(testValue)

	readVal := m.Read()
	if readVal != testValue {
		t.Errorf("Expected to read '%v', but got '%v'", testValue, readVal)
	}

	readValAgain := m.Read()
	if readValAgain != testValue {
		t.Errorf("Expected value to persist after read, but got '%v'", readValAgain)
	}
}

func TestTickDataMovement(t *testing.T) {
	m := NewShortTermMemory()
	testValue := 123
	m.SetVal(testValue)

	if len(m.A) != 1 || len(m.B) != 0 {
		t.Fatalf("Initial state incorrect. A len: %d, B len: %d", len(m.A), len(m.B))
	}
	if val := m.Read(); val != testValue {
		t.Fatalf("Initial read failed, got %v", val)
	}

	m.Tick()
	time.Sleep(10 * time.Millisecond)

	if len(m.A) != 0 || len(m.B) != 1 {
		t.Errorf("State after 1st tick incorrect. A len: %d, B len: %d", len(m.A), len(m.B))
	}
	if val := m.Read(); val != testValue {
		t.Errorf("Read after 1st tick failed, got %v", val)
	}

	m.Tick()
	time.Sleep(10 * time.Millisecond)

	if len(m.A) != 1 || len(m.B) != 0 {
		t.Errorf("State after 2nd tick incorrect. A len: %d, B len: %d", len(m.A), len(m.B))
	}
	if val := m.Read(); val != testValue {
		t.Errorf("Read after 2nd tick failed, got %v", val)
	}
}

func TestSetValOverrides(t *testing.T) {
	m := NewShortTermMemory()

	initialValue := "first"
	m.SetVal(initialValue)
	m.Tick()

	newValue := "second"
	m.SetVal(newValue)

	if val := m.Read(); val != newValue {
		t.Errorf("Expected to read the new value '%v', but got '%v'", newValue, val)
	}

	m.Tick()

	if val := m.Read(); val != initialValue {
		t.Errorf("Expected to read '%v' from channel A, but got '%v'", initialValue, val)
	}

	m.Tick()

	if val := m.Read(); val != newValue {
		t.Errorf("Expected to read '%v' from channel A, but got '%v'", newValue, val)
	}
}

func TestTickOnEmptyMemory(t *testing.T) {
	m := NewShortTermMemory()

	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Tick() panicked on empty memory: %v", r)
		}
	}()

	m.Tick()
	m.Tick()

	if val := m.Read(); val != nil {
		t.Errorf("Memory should still be empty, but read %v", val)
	}
}
