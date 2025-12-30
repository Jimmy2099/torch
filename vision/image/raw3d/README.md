# RAW3D

RAW3D is a 3D tensor built from stacked 2D images.

The third dimension represents:

1. Time (T) — video or image sequences
2. Depth (Z) — 3D data such as CAD, CT, or mesh

---

### Video (Time Series)

Image(X, Y):
Image₁ → Image₂ → Image₃ → Image₄

→ Tensor(X, Y, T)

---

### Spatial Volume

Slice(X, Y):
Slice₁ → Slice₂ → Slice₃ → Slice₄

→ Tensor(X, Y, Z)