using ILGPU;

namespace Raphael.Lazulite;

/// <summary>
/// A set of extensions for <see cref="LazuliteKernel{T}"/>s allowing easy kernel calling.
/// </summary>
public static class LazuliteKernelExtensions
{
    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    public static void Call<T1>(this LazuliteKernel<Action<Index1D, T1>> kernel, Index1D extent, T1 arg1)
        where T1 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1);

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    public static void Call<T1, T2>(this LazuliteKernel<Action<Index1D, T1, T2>> kernel, Index1D extent, T1 arg1, T2 arg2)
        where T1 : struct
        where T2 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2);

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <typeparam name="T3">The third argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    /// <param name="arg3">The third argument.</param>
    public static void Call<T1, T2, T3>(this LazuliteKernel<Action<Index1D, T1, T2, T3>> kernel, Index1D extent, T1 arg1, T2 arg2, T3 arg3)
        where T1 : struct
        where T2 : struct
        where T3 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3);

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <typeparam name="T3">The third argument's type.</typeparam>
    /// <typeparam name="T4">The fourth argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    /// <param name="arg3">The third argument.</param>
    /// <param name="arg4">The fourth argument.</param>
    public static void Call<T1, T2, T3, T4>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4>> kernel, Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4);

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <typeparam name="T3">The third argument's type.</typeparam>
    /// <typeparam name="T4">The fourth argument's type.</typeparam>
    /// <typeparam name="T5">The fifth argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    /// <param name="arg3">The third argument.</param>
    /// <param name="arg4">The fourth argument.</param>
    /// <param name="arg5">The fifth argument.</param>
    public static void Call<T1, T2, T3, T4, T5>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5>> kernel, Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4, arg5);

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <typeparam name="T3">The third argument's type.</typeparam>
    /// <typeparam name="T4">The fourth argument's type.</typeparam>
    /// <typeparam name="T5">The fifth argument's type.</typeparam>
    /// <typeparam name="T6">The sixth argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    /// <param name="arg3">The third argument.</param>
    /// <param name="arg4">The fourth argument.</param>
    /// <param name="arg5">The fifth argument.</param>
    /// <param name="arg6">The sixth argument.</param>
    public static void Call<T1, T2, T3, T4, T5, T6>(this LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6>> kernel,
        Index1D extent, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct =>
        (kernel._compiled ?? (kernel._compiled = kernel._lctx.Load(kernel))).Invoke(extent, arg1, arg2, arg3, arg4, arg5, arg6);

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <typeparam name="T3">The third argument's type.</typeparam>
    /// <typeparam name="T4">The fourth argument's type.</typeparam>
    /// <typeparam name="T5">The fifth argument's type.</typeparam>
    /// <typeparam name="T6">The sixth argument's type.</typeparam>
    /// <typeparam name="T7">The seventh argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    /// <param name="arg3">The third argument.</param>
    /// <param name="arg4">The fourth argument.</param>
    /// <param name="arg5">The fifth argument.</param>
    /// <param name="arg6">The sixth argument.</param>
    /// <param name="arg7">The seventh argument.</param>
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

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <typeparam name="T3">The third argument's type.</typeparam>
    /// <typeparam name="T4">The fourth argument's type.</typeparam>
    /// <typeparam name="T5">The fifth argument's type.</typeparam>
    /// <typeparam name="T6">The sixth argument's type.</typeparam>
    /// <typeparam name="T7">The seventh argument's type.</typeparam>
    /// <typeparam name="T8">The eighth argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    /// <param name="arg3">The third argument.</param>
    /// <param name="arg4">The fourth argument.</param>
    /// <param name="arg5">The fifth argument.</param>
    /// <param name="arg6">The sixth argument.</param>
    /// <param name="arg7">The seventh argument.</param>
    /// <param name="arg8">The eighth argument.</param>
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

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <typeparam name="T3">The third argument's type.</typeparam>
    /// <typeparam name="T4">The fourth argument's type.</typeparam>
    /// <typeparam name="T5">The fifth argument's type.</typeparam>
    /// <typeparam name="T6">The sixth argument's type.</typeparam>
    /// <typeparam name="T7">The seventh argument's type.</typeparam>
    /// <typeparam name="T8">The eighth argument's type.</typeparam>
    /// <typeparam name="T9">The ninth argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    /// <param name="arg3">The third argument.</param>
    /// <param name="arg4">The fourth argument.</param>
    /// <param name="arg5">The fifth argument.</param>
    /// <param name="arg6">The sixth argument.</param>
    /// <param name="arg7">The seventh argument.</param>
    /// <param name="arg8">The eighth argument.</param>
    /// <param name="arg9">The ninth argument.</param>
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

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <typeparam name="T3">The third argument's type.</typeparam>
    /// <typeparam name="T4">The fourth argument's type.</typeparam>
    /// <typeparam name="T5">The fifth argument's type.</typeparam>
    /// <typeparam name="T6">The sixth argument's type.</typeparam>
    /// <typeparam name="T7">The seventh argument's type.</typeparam>
    /// <typeparam name="T8">The eighth argument's type.</typeparam>
    /// <typeparam name="T9">The ninth argument's type.</typeparam>
    /// <typeparam name="T10">The tenth argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    /// <param name="arg3">The third argument.</param>
    /// <param name="arg4">The fourth argument.</param>
    /// <param name="arg5">The fifth argument.</param>
    /// <param name="arg6">The sixth argument.</param>
    /// <param name="arg7">The seventh argument.</param>
    /// <param name="arg8">The eighth argument.</param>
    /// <param name="arg9">The ninth argument.</param>
    /// <param name="arg10">The tenth argument.</param>
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

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <typeparam name="T3">The third argument's type.</typeparam>
    /// <typeparam name="T4">The fourth argument's type.</typeparam>
    /// <typeparam name="T5">The fifth argument's type.</typeparam>
    /// <typeparam name="T6">The sixth argument's type.</typeparam>
    /// <typeparam name="T7">The seventh argument's type.</typeparam>
    /// <typeparam name="T8">The eighth argument's type.</typeparam>
    /// <typeparam name="T9">The ninth argument's type.</typeparam>
    /// <typeparam name="T10">The tenth argument's type.</typeparam>
    /// <typeparam name="T11">The eleventh argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    /// <param name="arg3">The third argument.</param>
    /// <param name="arg4">The fourth argument.</param>
    /// <param name="arg5">The fifth argument.</param>
    /// <param name="arg6">The sixth argument.</param>
    /// <param name="arg7">The seventh argument.</param>
    /// <param name="arg8">The eighth argument.</param>
    /// <param name="arg9">The ninth argument.</param>
    /// <param name="arg10">The tenth argument.</param>
    /// <param name="arg11">The eleventh argument.</param>
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

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <typeparam name="T3">The third argument's type.</typeparam>
    /// <typeparam name="T4">The fourth argument's type.</typeparam>
    /// <typeparam name="T5">The fifth argument's type.</typeparam>
    /// <typeparam name="T6">The sixth argument's type.</typeparam>
    /// <typeparam name="T7">The seventh argument's type.</typeparam>
    /// <typeparam name="T8">The eighth argument's type.</typeparam>
    /// <typeparam name="T9">The ninth argument's type.</typeparam>
    /// <typeparam name="T10">The tenth argument's type.</typeparam>
    /// <typeparam name="T11">The eleventh argument's type.</typeparam>
    /// <typeparam name="T12">The twelfth argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    /// <param name="arg3">The third argument.</param>
    /// <param name="arg4">The fourth argument.</param>
    /// <param name="arg5">The fifth argument.</param>
    /// <param name="arg6">The sixth argument.</param>
    /// <param name="arg7">The seventh argument.</param>
    /// <param name="arg8">The eighth argument.</param>
    /// <param name="arg9">The ninth argument.</param>
    /// <param name="arg10">The tenth argument.</param>
    /// <param name="arg11">The eleventh argument.</param>
    /// <param name="arg12">The twelfth argument.</param>
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

    /// <summary>
    /// Calls the kernel, loading it if needed.
    /// </summary>
    /// <typeparam name="T1">The first argument's type.</typeparam>
    /// <typeparam name="T2">The second argument's type.</typeparam>
    /// <typeparam name="T3">The third argument's type.</typeparam>
    /// <typeparam name="T4">The fourth argument's type.</typeparam>
    /// <typeparam name="T5">The fifth argument's type.</typeparam>
    /// <typeparam name="T6">The sixth argument's type.</typeparam>
    /// <typeparam name="T7">The seventh argument's type.</typeparam>
    /// <typeparam name="T8">The eighth argument's type.</typeparam>
    /// <typeparam name="T9">The ninth argument's type.</typeparam>
    /// <typeparam name="T10">The tenth argument's type.</typeparam>
    /// <typeparam name="T11">The eleventh argument's type.</typeparam>
    /// <typeparam name="T12">The twelfth argument's type.</typeparam>
    /// <typeparam name="T13">The thirteenth argument's type.</typeparam>
    /// <param name="kernel">The kernel to call.</param>
    /// <param name="extent">The number of threads to run.</param>
    /// <param name="arg1">The first argument.</param>
    /// <param name="arg2">The second argument.</param>
    /// <param name="arg3">The third argument.</param>
    /// <param name="arg4">The fourth argument.</param>
    /// <param name="arg5">The fifth argument.</param>
    /// <param name="arg6">The sixth argument.</param>
    /// <param name="arg7">The seventh argument.</param>
    /// <param name="arg8">The eighth argument.</param>
    /// <param name="arg9">The ninth argument.</param>
    /// <param name="arg10">The tenth argument.</param>
    /// <param name="arg11">The eleventh argument.</param>
    /// <param name="arg12">The twelfth argument.</param>
    /// <param name="arg13">The thirteenth argument.</param>
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