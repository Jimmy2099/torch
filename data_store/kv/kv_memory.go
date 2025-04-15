package kv_memory

import "sync"

type Float32Pool struct {
	mu    sync.RWMutex
	pools map[string][]float32
}

func NewFloat32Pool() *Float32Pool {
	return &Float32Pool{
		pools: make(map[string][]float32),
	}
}

func (p *Float32Pool) Put(key string, data []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pools[key] = data
}

func (p *Float32Pool) Get(key string) ([]float32, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	data, exists := p.pools[key]
	return data, exists
}

func (p *Float32Pool) Delete(key string) []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	data := p.pools[key]
	delete(p.pools, key)
	return data
}

func (p *Float32Pool) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pools = make(map[string][]float32)
}
