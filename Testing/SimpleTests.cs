using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using Raphael.Lazulite;
using Raphael.Linalg32;

namespace Testing;

public class SimpleTests(LazuliteContext lctx)
{
    public void ElementwiseTest1(int size = -1)
    {
        if (size == -1) size = (int)Math.Pow(2, 20);

        var a = new float[size];
        var b = new float[size];
        
        using var ar = lctx.GetVector(size).Set(a);
        using var br = lctx.GetVector(size).Set(b);
        using var cr = lctx.GetVector(size);

        var sw = Stopwatch.StartNew();
        ar.Add(br, cr);
        sw.Stop();
        
        Console.WriteLine($"Elementwise add of size {size} took {sw.ElapsedMilliseconds}ms on {lctx.AcceleratorName}");
        
        sw.Restart();
        var c = a.Zip(b, (x, y) => x + y);
        sw.Stop();
        
        Console.WriteLine($"Elementwise add of size {size} took {sw.ElapsedMilliseconds}ms on CPU");
    }
    

    private float[] RandomVector(int size) => new float[size].Select(_ => Random.Shared.NextSingle()).ToArray();
}