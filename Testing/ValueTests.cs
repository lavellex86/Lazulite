using System.Diagnostics;
using ILGPU.Runtime;
using Raphael.Lazulite;
using Raphael.Lazulite.LinearAlgebra;

namespace Testing;

public static class ValueTests
{
    private static int _aidx = -1;
    
    public static void MathTest(bool gpu)
    {
        _aidx = Compute.RequestAccelerator(gpu);
        Console.WriteLine(Compute.IsGpuAccelerator(_aidx) ? $"GPU accelerator {_aidx}" : $"CPU accelerator {_aidx}");
        Stopwatch sw = new();

        using AcceleratedMatrix a = new(SimpleTests.RandomMatrix(10000, 10000), _aidx);
        using AcceleratedMatrix b = new(SimpleTests.RandomMatrix(10000, 10000), _aidx);
        using var c = a + b;
        
        sw.Start();
        float[,] matrix = c.ToHost();
        sw.Stop();
        Console.WriteLine($"ToHost elapsed time: {sw.ElapsedMilliseconds}");
        Console.WriteLine($"Matrix sum at (0, 0): {matrix[0, 0]}");

        using var d = a + b;
        sw.Restart();
        MatrixProxy proxy = d.ToProxy();
        sw.Stop();
        Console.WriteLine($"ToProxy elapsed time: {sw.ElapsedMilliseconds}");
        Console.WriteLine($"Matrix sum at (0, 0): {proxy[0, 0]}");
    }

    public static void OpsTest(bool gpu)
    {
        int aidx = Compute.RequestAccelerator(gpu);
        
        var a = Compute.FloatPool.Get(aidx, 4);
        var b = Compute.FloatPool.Get(aidx, 4);
        float[] realA = [1, 2, 3, 4];
        float[] realB = [3, 4, 5, 6];
        a.CopyFromCPU(realA);
        b.CopyFromCPU(realB);
        
        using var addBuffer = Compute.FloatPool.Get(aidx, 4);
        using var subBuffer = Compute.FloatPool.Get(aidx, 4);
        using var mulBuffer = Compute.FloatPool.Get(aidx, 4);
        using var divBuffer = Compute.FloatPool.Get(aidx, 4);
        using var maxBuffer = Compute.FloatPool.Get(aidx, 4);
        
        // Compute.Call(Compute.ElementwiseAddKernels, a.View, b.View, addBuffer.View);
        // Compute.Call(Compute.ElementwiseSubtractKernels, a.View, b.View, subBuffer.View);
        // Compute.Call(Compute.ElementwiseMultiplyKernels, a.View, b.View, mulBuffer.View);
        // Compute.Call(Compute.ElementwiseDivideKernels, a.View, b, divBuffer);
        // Compute.Call(Compute.ElementwiseMaxKernels, a.View, b, maxBuffer);
        LinearAlgebraSuite.Add(addBuffer, a, b);
        LinearAlgebraSuite.Subtract(subBuffer, a, b);
        LinearAlgebraSuite.ElementwiseMultiply(mulBuffer, a, b);
        LinearAlgebraSuite.Divide(divBuffer, a, b);
        LinearAlgebraSuite.Max(maxBuffer, a, b);
        
        Compute.Synchronize(aidx);
        
        Console.WriteLine($"a + b: {string.Join(',', addBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x + realB[i]))}");
        Console.WriteLine($"a - b: {string.Join(',', subBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x - realB[i]))}");
        Console.WriteLine($"a * b: {string.Join(',', mulBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x * realB[i]))}");
        Console.WriteLine($"a / b: {string.Join(',', divBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x / realB[i]))}");
        Console.WriteLine($"max(a, b): {string.Join(',', maxBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => Math.Max(x, realB[i])))}");

        using var expBuffer = Compute.FloatPool.Get(aidx, 4);
        using var logBuffer = Compute.FloatPool.Get(aidx, 4);
        using var sqrtBuffer = Compute.FloatPool.Get(aidx, 4);
        using var absBuffer = Compute.FloatPool.Get(aidx, 4);
        using var negateBuffer = Compute.FloatPool.Get(aidx, 4);
        
        // Compute.Call(Compute.ElementwiseExpKernels, a.View, expBuffer.View);
        // Compute.Call(Compute.ElementwiseLogKernels, a.View, logBuffer.View);
        // Compute.Call(Compute.ElementwiseSqrtKernels, a.View, sqrtBuffer.View);
        // Compute.Call(Compute.ElementwiseAbsKernels, a.View, absBuffer.View);
        // Compute.Call(Compute.ElementwiseNegateKernels, a.View, negateBuffer.View);
        LinearAlgebraSuite.Exp(expBuffer, a);
        LinearAlgebraSuite.Log(logBuffer, a);
        LinearAlgebraSuite.Sqrt(sqrtBuffer, a);
        LinearAlgebraSuite.Abs(absBuffer, a);
        LinearAlgebraSuite.Negate(negateBuffer, a);
        
        Compute.Synchronize(aidx);
        Console.WriteLine($"exp(a): {string.Join(',', expBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => (float)Math.Exp(x)))}");
        Console.WriteLine($"log(a): {string.Join(',', logBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => (float)Math.Log(x)))}");
        Console.WriteLine($"sqrt(a): {string.Join(',', sqrtBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => (float)Math.Sqrt(x)))}");
        Console.WriteLine($"abs(a): {string.Join(',', absBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(Math.Abs))}");
        Console.WriteLine($"-a: {string.Join(',', negateBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => -x))}");
        
        var dot = realA.Select((x, i) => x * realB[i]).Sum();
        using var dotBuffer = LinearAlgebraSuite.Dot(a, b);
        Compute.Synchronize(aidx);
        Console.WriteLine($"dot(a, b): {dot} vs {dotBuffer.GetAsArray1D()[0]}");
    }

    public static void MemoryTest(bool gpu)
    {
        int aidx = Compute.RequestAccelerator(gpu);
        
        var a = new AcceleratedScalar(1, aidx);
        var b = new AcceleratedScalar(2, aidx);
        var c = a + b;
        Console.WriteLine(a.Data.GetHashCode());
        Console.WriteLine(b.Data.GetHashCode());
        Console.WriteLine(c.Data.GetHashCode());
        a.Dispose();
        b.Dispose();
        c.Dispose();
        Compute.Synchronize(aidx);

        var d = new AcceleratedScalar(3, aidx);
        var e = d * c;
        Console.WriteLine(d.Data.GetHashCode());
        Console.WriteLine(e.Data.GetHashCode());
        d.Dispose();
        e.Dispose();
        Compute.Synchronize(aidx);

        var f = Compute.FloatPool.Get(aidx, 1);
        Console.WriteLine(f.GetAsArray1D()[0]);
        Console.WriteLine(f.GetHashCode());
    }
}