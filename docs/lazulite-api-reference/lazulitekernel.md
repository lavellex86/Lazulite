# LazuliteKernel

```csharp
public class LazuliteKernel<T>
```

Holds a loaded kernel under a context.

- `T`: The action type.

---

## Constructor

```csharp
public LazuliteKernel<T>(T action, LazuliteContext lctx)
```

- `action`: The action.
- `lctx`: The context under which this kernel runs.

---

## Extension Methods

Defined in `LazuliteKernelExtensions`
---

```csharp
public static void Call<T1>(
    this LazuliteKernel<Action<Index1D, T1>> kernel,
    Index1D extent,
    T1 arg1)
    where T1 : struct
```

Calls the kernel, loading it if needed. Overloads exist for kernels with 1 through 13 type parameters (`T1`–`T13`).

- `kernel`: The kernel to call.
- `extent`: The number of threads to run.
- `arg1`, `arg2`, ... `argN`: The first through Nth arguments.

