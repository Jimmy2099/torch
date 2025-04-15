package kv_memory

import "sync"

type MemoryPool struct {
	mu    sync.RWMutex
	pools map[string][]float32
}

func NewFloat32Pool() *MemoryPool {
	return &MemoryPool{
		pools: make(map[string][]float32),
	}
}

func (p *MemoryPool) Put(key string, data []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pools[key] = data
}

func (p *MemoryPool) Get(key string) ([]float32, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	data, exists := p.pools[key]
	return data, exists
}

func (p *MemoryPool) Delete(key string) []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	data := p.pools[key]
	delete(p.pools, key)
	return data
}

func (p *MemoryPool) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pools = make(map[string][]float32)
}
