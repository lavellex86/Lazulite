```csharp
public abstract class RemoteBase<TElement, THost> : IDisposable
    where TElement : unmanaged
    where THost : notnull
```
Represents an object on the compute device.
- `TElement`: The unmanaged data type stored on the compute device.
- `THost`: The class represented by the remote.

---

## Properties

```csharp
public MemoryBuffer1D<TElement, Stride1D.Dense> Buffer { get; }
```
The memory buffer of `TElement`s that holds the remote object.

---

```csharp
public bool Disposable { get; set; }
```
Whether this object is disposable. When `true`, it will be returned to the pool on `Dispose()`.

---

```csharp
public bool Disposed { get; }
```
Whether this remote has been disposed of.

---

```csharp
public int Length { get; }
```
The number of `TElement`s in the underlying memory buffer.

---

```csharp
public BufferPool<TElement> Pool { get; }
```
The buffer pool this object belongs to.

---

```csharp
public LazuliteContext Context { get; }
```
The Lazulite context over this remote.

---

## Methods

```csharp
public THost Get()
```
Synchronizes the context and returns the remote object as a `THost` instance, copying data from the device buffer to the host.

---

```csharp
public virtual RemoteBase<TElement, THost> Set(THost host)
```
Sets the remote object by converting `host` to a raw element array and copying it to the device buffer.
- `host`: The object to set the remote object to.

---

```csharp
public virtual RemoteBase<TElement, THost> Set(MemoryBuffer1D<TElement, Stride1D.Dense> source)
```
Sets the remote object by copying directly from another device memory buffer.
- `source`: The memory buffer to copy from.

---

```csharp
public void Dispose()
```
Returns the buffer to the pool if `Disposable` is `true`, invokes the dispose hook if set, and marks the remote as disposed.

---

```csharp
protected abstract THost ConvertToHost(TElement[] raw)
```
Converts a flattened device buffer into a `THost` object.
- `raw`: The flattened buffer to convert from.

---

```csharp
protected abstract TElement[] ConvertToRaw(THost host)
```
Converts a `THost` object into a flattened element array for device storage.
- `host`: The object to convert from.
