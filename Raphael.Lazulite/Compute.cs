using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace Raphael.Lazulite;

/// <summary>
/// The central accelerator management class.
/// </summary>
public static class Compute
{
    #region Properties & Fields
    /// <summary>
    /// A dictionary of available accelerators, formatted as {aidx: accelerator}.
    /// </summary>
    public static ConcurrentDictionary<int, Accelerator> Accelerators { get; } = [];
    
    /// <summary>
    /// A dictionary of accelerator indices, formatted as {accelerator name: aidx}.
    /// </summary>
    public static ConcurrentDictionary<string, int> AcceleratorIndices { get; } = [];
    
    #region Pools
    /// <summary>
    /// The default pool for <c>float</c> buffers.
    /// </summary>
    public static BufferPool<float> FloatPool { get; }
    /// <summary>
    /// The default pool for <c>double</c> buffers.
    /// </summary>
    public static BufferPool<double> DoublePool { get; }
    /// <summary>
    /// The default pool for <c>int</c> buffers.
    /// </summary>
    public static BufferPool<int> IntPool { get; }
    /// <summary>
    /// The default pool for <c>uint</c> buffers.
    /// </summary>
    public static BufferPool<uint> UnsignedIntPool { get; }
    /// <summary>
    /// The default pool for <c>long</c> buffers.
    /// </summary>
    public static BufferPool<long> LongPool { get; }
    /// <summary>
    /// The default pool for <c>ulong</c> buffers.
    /// </summary>
    public static BufferPool<ulong> UnsignedLongPool { get; }
    /// <summary>
    /// The default pool for <c>byte</c> buffers.
    /// </summary>
    public static BufferPool<byte> BytePool { get; }
    
    internal static List<IDisposable> BufferPoolHooks { get; } = [];
    #endregion

    private static Context _context { get; }
    private static bool _disposed;
    #endregion

    static Compute()
    {
        _context = Context.Create(b => b
            .Default()
            .EnableAlgorithms()
            .AllAccelerators());

        HashSet<(AcceleratorType, string, long)> seen = [];

        var aidx = 0;
        foreach (Device device in _context.Devices
                     .OrderBy(d => d.AcceleratorType switch { AcceleratorType.Cuda => 0, AcceleratorType.OpenCL => 1, AcceleratorType.CPU => 2, _ => 3 })
                     .ThenByDescending(d => d.MemorySize).Where(device => seen.Add((device.AcceleratorType, device.Name, device.MemorySize))))
        {
            Accelerators[aidx] = device.CreateAccelerator(_context);
            AcceleratorIndices[Accelerators[aidx].Name] = aidx;
            
            aidx++;
        }

        FloatPool = new();
        DoublePool = new();
        IntPool = new();
        UnsignedIntPool = new();
        LongPool = new();
        UnsignedLongPool = new();
        BytePool = new();
        
        AppDomain.CurrentDomain.ProcessExit += (_, _) => Dispose();
    }

    #region Management
    #region Synchronization
    /// <summary>
    /// Synchronizes the accelerator with the given index.
    /// </summary>
    public static void Synchronize(int aidx) => Accelerators[aidx].Synchronize();

    /// <summary>
    /// Synchronizes all accelerators.
    /// </summary>
    public static void SynchronizeAll()
    {
        for (int i = 0; i < Accelerators.Count; i++) Synchronize(i);
    }
    #endregion
    #region Accelerator Management
    /// <summary>
    /// Returns the best accelerator for the given requirements.
    /// </summary>
    /// <param name="requireGPU">Whether a GPU accelerator is required.</param>
    public static int RequestAccelerator(bool requireGPU = true)
    {
        Accelerator accelerator;
        
        if (requireGPU) accelerator = Accelerators.Values.FirstOrDefault(a => a is CudaAccelerator) ?? Accelerators.Values.First();
        else accelerator = Accelerators.Values.First();
        
        var aidx = GetAcceleratorIndex(accelerator);
        return aidx;
    }

    /// <summary>
    /// Returns the best CPU accelerator available.
    /// </summary>
    public static int RequestCPU()
    {
        var accelerator = Accelerators.Values.First(a => a is CPUAccelerator);
        return GetAcceleratorIndex(accelerator);    
    }

    /// <summary>
    /// Returns the best GPU accelerator available.
    /// </summary>
    public static int RequestGPU()
    {
        var accelerator = Accelerators.Values.First(a => a is CudaAccelerator or CLAccelerator);
        return GetAcceleratorIndex(accelerator);   
    }

    /// <summary>
    /// Returns the best accelerator available.
    /// </summary>
    public static int RequestOptimalAccelerator() => GetAcceleratorIndex(Accelerators.Values.First());
    #endregion
    #endregion
    #region Helpers
    /// <summary>
    /// Returns the index of the given accelerator.
    /// </summary>
    public static int GetAcceleratorIndex(Accelerator accelerator) => AcceleratorIndices[accelerator.Name];
    /// <summary>
    /// Whether the given accelerator is a GPU accelerator.
    /// </summary>
    public static bool IsGpuAccelerator(int aidx) => Accelerators[aidx] is CudaAccelerator;
    
    
    #region Calls
    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T>(int aidx, KernelStorage<Action<Index1D, T>> kernel, Index1D i, T a)
        where T : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2>(int aidx, KernelStorage<Action<Index1D, T1, T2>> kernel, Index1D i, T1 a, T2 b)
        where T1 : struct
        where T2 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3>> kernel, Index1D i, T1 a, T2 b, T3 c)
        where T1 : struct
        where T2 : struct
        where T3 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3, T4>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3, T4, T5>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3, T4, T5, T6>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3, T4, T5, T6, T7>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e,
        T6 f, T7 g)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d,
        T5 e, T6 f, T7 g, T8 h)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action); 
        kernel[aidx]!(i, a, b, c, d, e, f, g, h);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9>> kernel, Index1D i, T1 a, T2 b, T3 c,
        T4 d, T5 e, T6 f, T7 g, T8 h, T9 j) where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>> kernel, Index1D i, T1 a,
        T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 j, T10 k) where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j, k);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>> kernel,
        Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 j, T10 k, T11 l) where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct
        where T11 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j, k, l);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(int aidx,
        KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 j, T10 k, T11 l,
        T12 m) where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct
        where T11 : struct
        where T12 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j, k, l, m);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>(int aidx,
        KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 j, T10 k,
        T11 l, T12 m, T13 n) where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct
        where T11 : struct
        where T12 : struct
        where T13 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j, k, l, m, n);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments.
    /// </summary>
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>(int aidx,
        KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 j,
        T10 k, T11 l, T12 m, T13 n, T14 o) where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct
        where T11 : struct
        where T12 : struct
        where T13 : struct
        where T14 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j, k, l, m, n, o);
    }

    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData>(KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>>> kernel, ArrayView1D<TData, Stride1D.Dense> a) 
        where TData : unmanaged
        => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a);
    
    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T>(KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T>> kernel, ArrayView1D<TData, Stride1D.Dense> a, T b)
        where TData : unmanaged
        where T : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b);
    
    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2>(KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2>> kernel, ArrayView1D<TData, Stride1D.Dense> a, T1 b, T2 c)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c);
    
    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2, T3>(KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2, T3>> kernel, ArrayView1D<TData, Stride1D.Dense> a, T1 b, T2 c,
        T3 d)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct
        where T3 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d);

    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2, T3, T4>(KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2, T3, T4>> kernel, ArrayView1D<TData, Stride1D.Dense> a,
        T1 b, T2 c, T3 d, T4 e)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e);

    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2, T3, T4, T5>(KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2, T3, T4, T5>> kernel,
        ArrayView1D<TData, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f);

    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2, T3, T4, T5, T6>(KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2, T3, T4, T5, T6>> kernel,
        ArrayView1D<TData, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g);

    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2, T3, T4, T5, T6, T7>(KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7>> kernel,
        ArrayView1D<TData, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h);

    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2, T3, T4, T5, T6, T7, T8>(KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8>> kernel,
        ArrayView1D<TData, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i);

    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2, T3, T4, T5, T6, T7, T8, T9>(KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9>> kernel,
        ArrayView1D<TData, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i, j);

    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(
        KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>> kernel,
        ArrayView1D<TData, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i, j, k);

    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(
        KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>> kernel,
        ArrayView1D<TData, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k, T11 l)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct
        where T11 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i, j, k, l);

    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(
        KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>> kernel,
        ArrayView1D<TData, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k, T11 l, T12 m)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct
        where T11 : struct
        where T12 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i, j, k, l, m);

    /// <summary>
    /// Executes the given kernel with the given arguments, inferring the accelerator index and number of threads to launch from the first argument.
    /// </summary>
    public static void Call<TData, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>(
        KernelStorage<Action<Index1D, ArrayView1D<TData, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>> kernel,
        ArrayView1D<TData, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k, T11 l, T12 m, T13 n)
        where TData : unmanaged
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct
        where T11 : struct
        where T12 : struct
        where T13 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i, j, k, l, m, n);
    #endregion
    #endregion
    
    private static void Dispose()
    {
        if (_disposed) return;
        GC.WaitForPendingFinalizers();
        foreach (var pool in BufferPoolHooks) pool.Dispose();
        _context.Dispose();
        foreach (var accelerator in Accelerators.Values) accelerator.Dispose();
        _disposed = true;
    }
}