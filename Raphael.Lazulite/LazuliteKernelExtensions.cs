using ILGPU;

namespace Raphael.Lazulite;

public static class LazuliteKernelExtensions
{
    public static void Call<T1>(this LazuliteKernel<Action<Index1D, T1>> kernel, Index1D extent, T1 arg1)
        where T1 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1);

    public static void Call<T1, T2>(this LazuliteKernel<Action<Index1D, T1, T2>> kernel, Index1D extent, T1 arg1, T2 arg2)
        where T1 : struct
        where T2 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2);

    public static void Call<T1, T2, T3>(this LazuliteKernel<Action<Index1D, T1, T2, T3>> kernel, Index1D extent, T1 arg1, T2 arg2, T3 arg3)
        where T1 : struct
        where T2 : struct
        where T3 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3);

    public static void Call<T1, T2, T3, T4>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4>> kernel, Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4);

    public static void Call<T1, T2, T3, T4, T5>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5>> kernel, Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4, arg5);

    public static void Call<T1, T2, T3, T4, T5, T6>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6>> kernel, 
        Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4, arg5, arg6);

    public static void Call<T1, T2, T3, T4, T5, T6, T7>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7>> kernel, 
        Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4, arg5, arg6, arg7);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8>> kernel, 
        Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9>> kernel, 
        Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>> kernel, 
        Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>> kernel, 
        Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11)
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
        where T11 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>> kernel, 
        Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12)
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
        where T12 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>> kernel, 
        Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12, T13 arg13)
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
        where T13 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13);
}