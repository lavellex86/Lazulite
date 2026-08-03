# RemoteTensor

```csharp
public abstract class RemoteTensor<T> : RemoteBase<float, T>
    where T : notnull
```
Represents a tensor on the compute device. Derives from `RemoteBase<float, T`.
- `T`: The tensor type in `float` terms.

---
## Constructor
```csharp
public RemoteTensor<T>(
    FMB buffer,
    BufferPool<float> pool,
    int[] shape)
```
- `buffer`: The memory buffer underlying the tensor.
- `pool`: The buffer pool this tensor belongs to.
- `shape`: The shape of the tensor.

---
## Properties

```csharp
public int[] Shape { get; }
```
The shape of the tensor.

---

## Methods

```csharp
public abstract RemoteTensor<T> Create(FMB buffer, BufferPool<float> pool, int[] shape)
```
Creates a new `RemoteTensor<T>` from a buffer, pool, and shape.
- `buffer`: The memory buffer underlying the tensor.
- `pool`: The buffer pool this tensor belongs to.
- `shape`: The shape of the tensor.

---

```csharp
public RemoteTensor<T> Create(int[] shape, BufferPool<float> pool, bool cleared = false)
```
Creates a new `RemoteTensor<T>` from a pool and shape, allocating a buffer whose size is the product of all shape dimensions.
- `shape`: The shape of the tensor.
- `pool`: The buffer pool to allocate from.
- `cleared`: Whether the allocated buffer should be zero-initialized.

---

```csharp
public RemoteTensor<T> Create(int[] shape, bool cleared = false)
```
Creates a new `RemoteTensor<T>` from a shape, using the current instance's pool.
- `shape`: The shape of the tensor.
- `cleared`: Whether the allocated buffer should be zero-initialized.

---

```csharp
public RemoteTensor<T> Create(bool cleared = false)
```
Creates a new `RemoteTensor<T>` with the same shape as the current instance.
- `cleared`: Whether the allocated buffer should be zero-initialized.

---

## Conversions

```csharp
public static implicit operator FAV(RemoteTensor<T> tensor)
```
Converts the tensor to an ILGPU array view object.

---

```csharp
public static implicit operator FMB(RemoteTensor<T> tensor)
```
Converts the tensor to an ILGPU memory buffer object.