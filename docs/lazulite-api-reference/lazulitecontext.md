# LazuliteContext

```csharp
public class LazuliteContext : IDisposable
```

Holds all information regarding the compute device and manages extension libraries.

---

## Constructor

```csharp
public LazuliteContext(
    bool gpu = true,
    OptimizationLevel optimization = OptimizationLevel.Release,
    Accelerator? accelerator = null)
```

- `gpu`: Whether to look for a GPU accelerator or a CPU accelerator.
- `optimization`: The level of optimization ILGPU should use to compile kernels. Release mode by default.
- `accelerator`: The ILGPU accelerator to use.

---

## Properties

```csharp
public Accelerator Accelerator { get; }
```

The ILGPU accelerator underlying Lazulite.

---

```csharp
public List<Action> DisposeHooks { get; }
```

A set of actions to run upon disposal.

---

```csharp
public string AcceleratorName { get; }
```

The name of the compute device.

---

## Methods

```csharp
public void Synchronize()
```

Synchronizes the runtime with the compute device.

---

```csharp
public void Dispose()
```

Calls all `DisposeHooks` and disposes of the ILGPU accelerator.

---

## Extension Methods

```csharp
public static Action<Index1D, T1> Load<T1>(
    this LazuliteContext lctx,
    LazuliteKernel<Action<Index1D, T1>> kernel)
    where T1 : struct
```

Loads a kernel onto the compute device. Overloads exist for kernels with 1 through 14 type parameters (`T1`–`T14`), each constrained to `struct`.

- `kernel`: The kernel to load.