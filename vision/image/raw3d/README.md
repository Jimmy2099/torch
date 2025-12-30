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

### RAW3D(X,Y,Z) to Tensor(X*C, Y*R)
RAW3D(X, Y, Z) can be tiled into a grid Tensor(X * C, Y * R) where C (columns) * R (rows) = Z.  
*Example: 100 images of 50x50 (Z=100) -> 10x10 grid -> Tensor(500, 500)*