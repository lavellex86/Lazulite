```csharp
public class RemoteScalar : RemoteTensor<float>
```
Represents a scalar on the compute device. Derives from `RemoteTensor<float>`.

---
## Constructor
```csharp
public RemoteScalar(FMB buffer, BufferPool<float> pool)
```
- `buffer`: The memory buffer underlying the scalar.
- `pool`: The buffer pool this scalar belongs to.