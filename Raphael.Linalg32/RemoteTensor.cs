using System.Runtime.CompilerServices;
using ILGPU.Runtime;
using Raphael.Lazulite;

namespace Raphael.Linalg32;

public abstract class RemoteTensor<T>(FMB buffer, BufferPool<float> pool, int[] shape) : RemoteBase<float, T>(buffer, pool), IRemoteTensor 
    where T : notnull
{
    public int[] Shape { get; } = shape;

    public abstract RemoteTensor<T> Create(FMB buffer, BufferPool<float> pool, int[] shape);
    public RemoteTensor<T> Create(int[] shape, BufferPool<float> pool) => Create(pool.Get(shape.Aggregate((a, b) => a * b)), pool, shape);
    public RemoteTensor<T> Create(int[] shape) => Create(shape, Pool);

    public override RemoteTensor<T> Set(T host)
    {
        Context.Synchronize();
        Buffer.CopyFromCPU(ConvertToRaw(host));
        return this;
    }

    public static implicit operator FAV(RemoteTensor<T> tensor) => tensor.Buffer;
    public static implicit operator FMB(RemoteTensor<T> tensor) => tensor.Buffer;

    IRemoteTensor IRemoteTensor.Create(FMB buffer, BufferPool<float> pool, int[] shape) => Create(buffer, pool, shape);
    IRemoteTensor IRemoteTensor.Create(int[] shape, BufferPool<float> pool) => Create(shape, pool);
    IRemoteTensor IRemoteTensor.Create(int[] shape) => Create(shape);
}

public class RemoteScalar(FMB buffer, BufferPool<float> pool) : RemoteTensor<float>(buffer, pool, [])
{
    public override float ConvertToHost(float[] raw) => raw[0];
    public override float[] ConvertToRaw(float host) => [host];

    public override RemoteScalar Create(FMB buffer, BufferPool<float> pool, int[] shape) => new(buffer, pool);
}

public class RemoteVector(FMB buffer, BufferPool<float> pool) : RemoteTensor<float[]>(buffer, pool, [buffer.IntExtent])
{
    public override float[] ConvertToHost(float[] raw) => raw;
    public override float[] ConvertToRaw(float[] host) => host;

    public override RemoteVector Create(FMB buffer, BufferPool<float> pool, int[] shape) => new(buffer, pool);
}

    public class RemoteMatrix(FMB buffer, BufferPool<float> pool, int m0) : RemoteTensor<float[,]>(buffer, pool, [m0, buffer.IntExtent / m0])
    {
        public override float[,] ConvertToHost(float[] raw)
        {
            var matrix = new float[Shape[0], Shape[1]];
            for (int i = 0; i < Shape[0]; i++)
            for (int j = 0; i < Shape[1]; i++)
                matrix[i, j] = raw[i * Shape[0] + j];
            return matrix;
        }

        public override float[] ConvertToRaw(float[,] host)
        {
            var raw = new float[host.GetLength(0) * host.GetLength(1)];
            for (int i = 0; i < Shape[0]; i++)
            for (int j = 0; i < Shape[1]; i++)
                raw[i * Shape[0] + j] = host[i, j];
            return raw;
        }

        public override RemoteMatrix Create(FMB buffer, BufferPool<float> pool, int[] shape) => new(buffer, pool, shape[0]);
    }