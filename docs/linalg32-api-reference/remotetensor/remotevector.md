# RemoteVector

```csharp
public class RemoteVector : RemoteTensor<float[]>
```
Represents a vector on the compute device. Derives from `RemoteTensor<float[]>`.

---
## Constructor
```csharp
public RemoteVector(FMB buffer, BufferPool<float> pool)
```
- `buffer`: The memory buffer underlying the vector.
- `pool`: The buffer pool this vector belongs to.
