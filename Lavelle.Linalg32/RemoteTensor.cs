using Lavelle.Lazulite;
using System.Numerics;

namespace Lavelle.Linalg32;

/// <summary>
/// Represents a tensor on the compute device.
/// </summary>
/// <typeparam name="T">The tensor type in <see cref="float"/> terms.</typeparam>
/// <param name="buffer">The memory buffer underlying the tensor.</param>
/// <param name="pool">The buffer pool this tensor belongs to.</param>
/// <param name="shape">The shape of the tensor.</param>
public abstract class RemoteTensor<T>(FMB buffer, BufferPool<float> pool, int[] shape) : RemoteBase<float, T>(buffer, pool)
    where T : notnull
{
    /// <summary>
    /// The shape of the tensor.
    /// </summary>
    public int[] Shape { get; } = shape;

    /// <summary>
    /// Creates a new <see cref="RemoteTensor{T}"/> object from a buffer, pool, and shape.
    /// </summary>
    public abstract RemoteTensor<T> Create(FMB buffer, BufferPool<float> pool, int[] shape);
    /// <summary>
    /// Creates a new <see cref="RemoteTensor{T}"/> object from a pool and shape.
    /// </summary>
    public RemoteTensor<T> Create(int[] shape, BufferPool<float> pool, bool cleared = false) => 
        Create(pool.Get(shape.Aggregate((a, b) => a * b), cleared), pool, shape);
    /// <summary>
    /// Creates a new <see cref="RemoteTensor{T}"/> object from a shape.
    /// </summary>
    public RemoteTensor<T> Create(int[] shape, bool cleared = false) => Create(shape, Pool, cleared);
    /// <summary>
    /// Creates a new <see cref="RemoteTensor{T}"/> object.
    /// </summary>
    public RemoteTensor<T> Create(bool cleared = false) => Create(Shape, false);

    /// <summary>
    /// Converts the tensor to an ILGPU array view object.
    /// </summary>
    public static implicit operator FAV(RemoteTensor<T> tensor) => tensor.Buffer;
    /// <summary>
    /// Converts the tensor to an ILGPU memory buffer object.
    /// </summary>
    public static implicit operator FMB(RemoteTensor<T> tensor) => tensor.Buffer;
}

/// <summary>
/// Represents a scalar on the compute device.
/// </summary>
/// <param name="buffer">The memory buffer underlying the scalar.</param>
/// <param name="pool">The buffer pool this scalar belongs to.</param>
public class RemoteScalar(FMB buffer, BufferPool<float> pool) : RemoteTensor<float>(buffer, pool, [])
{
    /// <summary>
    /// Converts a flattened buffer to a target object.
    /// </summary>
    /// <param name="raw">The flattened buffer to convert from.</param>
    protected override float ConvertToHost(float[] raw) => raw[0];
    /// <summary>
    /// Converts an object into a flattened buffer.
    /// </summary>
    /// <param name="host">The object to convert from.</param>
    protected override float[] ConvertToRaw(float host) => [host];

    /// <summary>
    /// Creates a new <see cref="RemoteTensor{T}"/> object from a buffer, pool, and shape.
    /// </summary>
    public override RemoteScalar Create(FMB buffer, BufferPool<float> pool, int[] shape) => new(buffer, pool);
}


/// <summary>
/// Represents a vector on the compute device.
/// </summary>
/// <param name="buffer">The memory buffer underlying the vector.</param>
/// <param name="pool">The buffer pool this vector belongs to.</param>
public class RemoteVector(FMB buffer, BufferPool<float> pool) : RemoteTensor<float[]>(buffer, pool, [buffer.IntExtent])
{
    /// <summary>
    /// Converts a flattened buffer to a target object.
    /// </summary>
    /// <param name="raw">The flattened buffer to convert from.</param>
    protected override float[] ConvertToHost(float[] raw) => raw;
    /// <summary>
    /// Converts an object into a flattened buffer.
    /// </summary>
    /// <param name="host">The object to convert from.</param>
    protected override float[] ConvertToRaw(float[] host) => host;

    /// <summary>
    /// Creates a new <see cref="RemoteVector"/> from a buffer, pool, and shape.
    /// </summary>
    public override RemoteVector Create(FMB buffer, BufferPool<float> pool, int[] shape) => new(buffer, pool);
}

/// <summary>
/// Represents a matrix on the compute device, stored in row-major order.
/// </summary>
/// <param name="buffer">The memory buffer underlying the matrix.</param>
/// <param name="pool">The buffer pool this matrix belongs to.</param>
/// <param name="m0">The number of rows. The number of columns is inferred as <c>buffer.IntExtent / m0</c>.</param>
public class RemoteMatrix(FMB buffer, BufferPool<float> pool, int m0) : RemoteTensor<float[,]>(buffer, pool, [m0, buffer.IntExtent / m0])
{
    /// <summary>
    /// Converts a flattened buffer to a target object.
    /// </summary>
    /// <param name="raw">The flattened buffer to convert from.</param>
    protected override float[,] ConvertToHost(float[] raw)
    {
        var matrix = new float[Shape[0], Shape[1]];
        for (int i = 0; i < Shape[0]; i++)
            for (int j = 0; j < Shape[1]; j++)
                matrix[i, j] = raw[i * Shape[1] + j];
        return matrix;
    }

    /// <summary>
    /// Converts an object into a flattened buffer.
    /// </summary>
    /// <param name="host">The object to convert from.</param>
    protected override float[] ConvertToRaw(float[,] host)
    {
        var raw = new float[host.GetLength(0) * host.GetLength(1)];
        for (int i = 0; i < Shape[0]; i++)
            for (int j = 0; j < Shape[1]; j++)
                raw[i * Shape[1] + j] = host[i, j];
        return raw;
    }

    /// <summary>
    /// Creates a new <see cref="RemoteMatrix"/> from a buffer, pool, and shape.
    /// </summary>
    public override RemoteMatrix Create(FMB buffer, BufferPool<float> pool, int[] shape) => new(buffer, pool, shape[0]);
}