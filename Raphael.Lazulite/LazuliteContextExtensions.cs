using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

/// <summary>
/// A set of extensions for the <see cref="LazuliteContext"/> allowing easy kernel loading.
/// </summary>
public static class LazuliteContextExtensions
{
    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1> Load<T1>(this LazuliteContext lctx, LazuliteKernel<Action<Index1D, T1>> kernel)
        where T1 : struct =>
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2> Load<T1, T2>(this LazuliteContext lctx, LazuliteKernel<Action<Index1D, T1, T2>> kernel)
        where T1 : struct
        where T2 : struct =>
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3> Load<T1, T2, T3>(this LazuliteContext lctx, LazuliteKernel<Action<Index1D, T1, T2, T3>> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct =>
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T4">The fourth argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3, T4> Load<T1, T2, T3, T4>(this LazuliteContext lctx, LazuliteKernel<Action<Index1D, T1, T2, T3, T4>> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct =>
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T4">The fourth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T5">The fifth argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3, T4, T5> Load<T1, T2, T3, T4, T5>(this LazuliteContext lctx, LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5>> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct =>
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T4">The fourth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T5">The fifth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T6">The sixth argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3, T4, T5, T6> Load<T1, T2, T3, T4, T5, T6>(this LazuliteContext lctx, LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6>> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct =>
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T4">The fourth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T5">The fifth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T6">The sixth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T7">The seventh argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7> Load<T1, T2, T3, T4, T5, T6, T7>(this LazuliteContext lctx,
        LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7>> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct =>
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T4">The fourth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T5">The fifth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T6">The sixth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T7">The seventh argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T8">The eighth argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8> Load<T1, T2, T3, T4, T5, T6, T7, T8>(this LazuliteContext lctx,
        LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8>> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct =>
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T4">The fourth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T5">The fifth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T6">The sixth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T7">The seventh argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T8">The eighth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T9">The ninth argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9> Load<T1, T2, T3, T4, T5, T6, T7, T8, T9>(this LazuliteContext lctx,
        LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9>> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct =>
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T4">The fourth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T5">The fifth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T6">The sixth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T7">The seventh argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T8">The eighth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T9">The ninth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T10">The tenth argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> Load<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(this LazuliteContext lctx,
        LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>> kernel)
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
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T4">The fourth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T5">The fifth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T6">The sixth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T7">The seventh argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T8">The eighth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T9">The ninth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T10">The tenth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T11">The eleventh argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> Load<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(this LazuliteContext lctx,
        LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>> kernel)
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
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T4">The fourth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T5">The fifth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T6">The sixth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T7">The seventh argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T8">The eighth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T9">The ninth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T10">The tenth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T11">The eleventh argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T12">The twelfth argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> Load<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(this LazuliteContext lctx,
        LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>> kernel)
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
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T4">The fourth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T5">The fifth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T6">The sixth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T7">The seventh argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T8">The eighth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T9">The ninth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T10">The tenth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T11">The eleventh argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T12">The twelfth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T13">The thirteenth argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13> Load<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>(this LazuliteContext lctx,
        LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>> kernel)
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
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);

    /// <summary>
    /// Loads a kernel onto the compute device.
    /// </summary>
    /// <typeparam name="T1">The first argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T2">The second argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T3">The third argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T4">The fourth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T5">The fifth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T6">The sixth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T7">The seventh argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T8">The eighth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T9">The ninth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T10">The tenth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T11">The eleventh argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T12">The twelfth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T13">The thirteenth argument's type (cannot be a class).</typeparam>
    /// <typeparam name="T14">The fourteenth argument's type (cannot be a class).</typeparam>
    /// <param name="lctx">The context to load with.</param>
    /// <param name="kernel">The kernel to load.</param>
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14> Load<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>(this LazuliteContext lctx,
        LazuliteKernel<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>> kernel)
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
        where T13 : struct
        where T14 : struct =>
        lctx.Accelerator.LoadAutoGroupedStreamKernel(kernel._action);
}