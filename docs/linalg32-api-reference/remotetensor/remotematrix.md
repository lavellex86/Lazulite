# RemoteMatrix

```csharp
public class RemoteMatrix : RemoteTensor<float[,]>
```
Represents a matrix on the compute device, stored in row-major order. Derives from `RemoteTensor<float[,]>`.

---
## Constructor
```csharp
public RemoteMatrix(FMB buffer, BufferPool<float> pool, int m0)
```
- `buffer`: The memory buffer underlying the matrix.
- `pool`: The buffer pool this matrix belongs to.
- `m0`: The number of rows. The number of columns is inferred as `buffer.IntExtent / m0`.