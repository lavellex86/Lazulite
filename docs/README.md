```csharp
public class BufferPool<T> where T : unmanaged
```
A pool of reusable device memory buffers.
- `T`: The buffer element type to pool.

---

## Methods

```csharp
public void Return(MemoryBuffer1D<T, Stride1D.Dense> buffer)
```
Returns a buffer to the pool.
- `buffer`: The buffer to return.

---

```csharp
public void Return(params IEnumerable<MemoryBuffer1D<T, Stride1D.Dense>> buffers)
```
Returns a set of buffers to the pool.
- `buffers`: The buffers to return.

---

```csharp
public MemoryBuffer1D<T, Stride1D.Dense> Get(long length, bool cleared = false)
```
Retrieves a buffer of the given length from the pool, allocating a new one if none is available.
- `length`: The length of the buffer.
- `cleared`: Whether to zero the buffer before returning it.

---

```csharp
public void Dispose()
```
Disposes of all buffers currently held in the pool.
