using System.Diagnostics;
using Raphael.Lazulite;
using Raphael.Linalg32;

namespace Testing;

public class SimpleTests(LazuliteContext lctx)
{
    // ========== Tolerance for floating-point comparisons ==========
    private const float EPSILON = 1e-5f;

    // ========== UTILITY METHODS ==========
    
    private float[] RandomVector(int size) => new float[size].Select(_ => Random.Shared.NextSingle()).ToArray();
    private float[] RandomVector(int size, float min, float max) => 
        new float[size].Select(_ => min + Random.Shared.NextSingle() * (max - min)).ToArray();
    
    private float[,] RandomMatrix(int rows, int cols)
    {
        var matrix = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = Random.Shared.NextSingle();
        return matrix;
    }
    
    private bool VectorsEqual(float[] a, float[] b) =>
        a.Length == b.Length && a.Zip(b, (x, y) => Math.Abs(x - y) < EPSILON).All(x => x);
    
    private bool MatricesEqual(float[,] a, float[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1)) return false;
        for (int i = 0; i < a.GetLength(0); i++)
            for (int j = 0; j < a.GetLength(1); j++)
                if (Math.Abs(a[i, j] - b[i, j]) >= EPSILON) return false;
        return true;
    }

    // ========== ARITHMETIC OPERATIONS TESTS ==========
    
    public void TestAddition()
    {
        Console.WriteLine("\n=== Testing Addition ===");
        int[] sizes = { 1000, 10000, 100000, 1000000 };

        foreach (int size in sizes)
        {
            var a = RandomVector(size);
            var b = RandomVector(size);

            using var ar = lctx.GetVector(size).Set(a);
            using var br = lctx.GetVector(size).Set(b);
            using var cr = lctx.GetVector(size);

            var sw = Stopwatch.StartNew();
            ar.Add(br, cr);
            sw.Stop();
            var gpuResult = cr.ToHost();

            var cpuResult = a.Zip(b, (x, y) => x + y).ToArray();

            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Size {size}: {sw.ElapsedMilliseconds}ms - {(correct ? "✓ PASS" : "✗ FAIL")}");
        }
    }

    public void TestSubtraction()
    {
        Console.WriteLine("\n=== Testing Subtraction ===");
        int[] sizes = { 1000, 10000, 100000, 1000000 };

        foreach (int size in sizes)
        {
            var a = RandomVector(size);
            var b = RandomVector(size);

            using var ar = lctx.GetVector(size).Set(a);
            using var br = lctx.GetVector(size).Set(b);
            using var cr = lctx.GetVector(size);

            var sw = Stopwatch.StartNew();
            ar.Subtract(br, cr);
            sw.Stop();
            var gpuResult = cr.ToHost();

            var cpuResult = a.Zip(b, (x, y) => x - y).ToArray();

            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Size {size}: {sw.ElapsedMilliseconds}ms - {(correct ? "✓ PASS" : "✗ FAIL")}");
        }
    }

    public void TestMultiplication()
    {
        Console.WriteLine("\n=== Testing Multiplication ===");
        int[] sizes = { 1000, 10000, 100000, 1000000 };

        foreach (int size in sizes)
        {
            var a = RandomVector(size);
            var b = RandomVector(size);

            using var ar = lctx.GetVector(size).Set(a);
            using var br = lctx.GetVector(size).Set(b);
            using var cr = lctx.GetVector(size);

            var sw = Stopwatch.StartNew();
            ar.Multiply(br, cr);
            sw.Stop();
            var gpuResult = cr.ToHost();

            var cpuResult = a.Zip(b, (x, y) => x * y).ToArray();

            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Size {size}: {sw.ElapsedMilliseconds}ms - {(correct ? "✓ PASS" : "✗ FAIL")}");
        }
    }

    public void TestDivision()
    {
        Console.WriteLine("\n=== Testing Division ===");
        int[] sizes = { 1000, 10000, 100000, 1000000 };

        foreach (int size in sizes)
        {
            var a = RandomVector(size, 0.1f, 10f);
            var b = RandomVector(size, 0.1f, 10f);

            using var ar = lctx.GetVector(size).Set(a);
            using var br = lctx.GetVector(size).Set(b);
            using var cr = lctx.GetVector(size);

            var sw = Stopwatch.StartNew();
            ar.Divide(br, cr);
            sw.Stop();
            var gpuResult = cr.ToHost();

            var cpuResult = a.Zip(b, (x, y) => x / y).ToArray();

            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Size {size}: {sw.ElapsedMilliseconds}ms - {(correct ? "✓ PASS" : "✗ FAIL")}");
        }
    }

    public void TestScalarOperations()
    {
        Console.WriteLine("\n=== Testing Scalar Operations ===");
        int size = 100000;
        var a = RandomVector(size);
        float scalar = 2.5f;

        using var ar = lctx.GetVector(size).Set(a);
        using var addResult = lctx.GetVector(size);
        using var mulResult = lctx.GetVector(size);
        using var subResult = lctx.GetVector(size);
        using var divResult = lctx.GetVector(size);

        var sw = Stopwatch.StartNew();
        ar.AddScalar(scalar, addResult);
        sw.Stop();
        var gpuAdd = addResult.ToHost();
        var cpuAdd = a.Select(x => x + scalar).ToArray();
        bool addCorrect = VectorsEqual(gpuAdd, cpuAdd);
        Console.WriteLine($"AddScalar: {sw.ElapsedMilliseconds}ms - {(addCorrect ? "✓ PASS" : "✗ FAIL")}");

        sw.Restart();
        ar.SubtractScalar(scalar, subResult);
        sw.Stop();
        var gpuSub = subResult.ToHost();
        var cpuSub = a.Select(x => x - scalar).ToArray();
        bool subCorrect = VectorsEqual(gpuSub, cpuSub);
        Console.WriteLine($"SubtractScalar: {sw.ElapsedMilliseconds}ms - {(subCorrect ? "✓ PASS" : "✗ FAIL")}");

        sw.Restart();
        ar.MultiplyScalar(scalar, mulResult);
        sw.Stop();
        var gpuMul = mulResult.ToHost();
        var cpuMul = a.Select(x => x * scalar).ToArray();
        bool mulCorrect = VectorsEqual(gpuMul, cpuMul);
        Console.WriteLine($"MultiplyScalar: {sw.ElapsedMilliseconds}ms - {(mulCorrect ? "✓ PASS" : "✗ FAIL")}");

        sw.Restart();
        ar.DivideScalar(scalar, divResult);
        sw.Stop();
        var gpuDiv = divResult.ToHost();
        var cpuDiv = a.Select(x => x / scalar).ToArray();
        bool divCorrect = VectorsEqual(gpuDiv, cpuDiv);
        Console.WriteLine($"DivideScalar: {sw.ElapsedMilliseconds}ms - {(divCorrect ? "✓ PASS" : "✗ FAIL")}");
    }

    // ========== MATHEMATICAL FUNCTIONS TESTS ==========
    
    public void TestTranscendentalFunctions()
    {
        Console.WriteLine("\n=== Testing Transcendental Functions ===");
        int size = 10000;
        var a = RandomVector(size, 0.1f, 3.14f); // Keep in reasonable range for trig functions

        using var ar = lctx.GetVector(size).Set(a);

        // Test Exp
        using (var result = lctx.GetVector(size))
        {
            ar.Exp(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Exp(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Exp: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        var a_positive = RandomVector(size, 0.1f, 100f);
        using var ar_pos = lctx.GetVector(size).Set(a_positive);

        // Test Log
        using (var result = lctx.GetVector(size))
        {
            ar_pos.Log(result);
            var gpuResult = result.ToHost();
            var cpuResult = a_positive.Select(x => (float)Math.Log(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Log: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Log10
        using (var result = lctx.GetVector(size))
        {
            ar_pos.Log10(result);
            var gpuResult = result.ToHost();
            var cpuResult = a_positive.Select(x => (float)Math.Log10(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Log10: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Log2
        using (var result = lctx.GetVector(size))
        {
            ar_pos.Log2(result);
            var gpuResult = result.ToHost();
            var cpuResult = a_positive.Select(x => (float)Math.Log(x, 2)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Log2: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Sqrt
        using (var result = lctx.GetVector(size))
        {
            ar_pos.Sqrt(result);
            var gpuResult = result.ToHost();
            var cpuResult = a_positive.Select(x => (float)Math.Sqrt(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Sqrt: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Sin
        using (var result = lctx.GetVector(size))
        {
            ar.Sin(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Sin(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Sin: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Cos
        using (var result = lctx.GetVector(size))
        {
            ar.Cos(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Cos(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Cos: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Tan
        using (var result = lctx.GetVector(size))
        {
            ar.Tan(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Tan(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Tan: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }
    }

    public void TestHyperbolicFunctions()
    {
        Console.WriteLine("\n=== Testing Hyperbolic Functions ===");
        int size = 10000;
        var a = RandomVector(size, -2f, 2f); // Keep in reasonable range

        using var ar = lctx.GetVector(size).Set(a);

        // Test Sinh
        using (var result = lctx.GetVector(size))
        {
            ar.Sinh(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Sinh(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Sinh: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Cosh
        using (var result = lctx.GetVector(size))
        {
            ar.Cosh(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Cosh(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Cosh: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Tanh
        using (var result = lctx.GetVector(size))
        {
            ar.Tanh(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Tanh(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Tanh: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }
    }

    public void TestRoundingFunctions()
    {
        Console.WriteLine("\n=== Testing Rounding Functions ===");
        int size = 10000;
        var a = RandomVector(size, -100f, 100f);

        using var ar = lctx.GetVector(size).Set(a);

        // Test Abs
        using (var result = lctx.GetVector(size))
        {
            ar.Abs(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => Math.Abs(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Abs: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Floor
        using (var result = lctx.GetVector(size))
        {
            ar.Floor(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Floor(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Floor: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Ceiling
        using (var result = lctx.GetVector(size))
        {
            ar.Ceiling(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Ceiling(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Ceiling: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Round
        using (var result = lctx.GetVector(size))
        {
            ar.Round(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Round(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Round: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Truncate
        using (var result = lctx.GetVector(size))
        {
            ar.Truncate(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Truncate(x)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Truncate: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Sign
        using (var result = lctx.GetVector(size))
        {
            ar.Sign(result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => Math.Sign(x)).ToArray();
            bool correct = gpuResult.Zip(cpuResult, (g, c) => Math.Abs(g - c) < EPSILON).All(x => x);
            Console.WriteLine($"Sign: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }
    }

    // ========== MIN/MAX TESTS ==========
    
    public void TestMinMax()
    {
        Console.WriteLine("\n=== Testing Min/Max Operations ===");
        int size = 10000;
        var a = RandomVector(size, -100f, 100f);
        var b = RandomVector(size, -100f, 100f);

        using var ar = lctx.GetVector(size).Set(a);
        using var br = lctx.GetVector(size).Set(b);

        // Test Min
        using (var result = lctx.GetVector(size))
        {
            ar.Min(br, result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Zip(b, (x, y) => Math.Min(x, y)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Min: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test Max
        using (var result = lctx.GetVector(size))
        {
            ar.Max(br, result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Zip(b, (x, y) => Math.Max(x, y)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Max: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test MinScalar
        float scalar = 5.0f;
        using (var result = lctx.GetVector(size))
        {
            ar.MinScalar(scalar, result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => Math.Min(x, scalar)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"MinScalar: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test MaxScalar
        using (var result = lctx.GetVector(size))
        {
            ar.MaxScalar(scalar, result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => Math.Max(x, scalar)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"MaxScalar: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }
    }

    public void TestPower()
    {
        Console.WriteLine("\n=== Testing Power Functions ===");
        int size = 10000;
        var a = RandomVector(size, 0.1f, 10f);
        var b = RandomVector(size, 0.5f, 3f);

        using var ar = lctx.GetVector(size).Set(a);
        using var br = lctx.GetVector(size).Set(b);

        // Test Pow
        using (var result = lctx.GetVector(size))
        {
            ar.Pow(br, result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Zip(b, (x, y) => (float)Math.Pow(x, y)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"Pow: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }

        // Test PowScalar
        float scalar = 2.0f;
        using (var result = lctx.GetVector(size))
        {
            ar.PowScalar(scalar, result);
            var gpuResult = result.ToHost();
            var cpuResult = a.Select(x => (float)Math.Pow(x, scalar)).ToArray();
            bool correct = VectorsEqual(gpuResult, cpuResult);
            Console.WriteLine($"PowScalar: {(correct ? "✓ PASS" : "✗ FAIL")}");
        }
    }

    // ========== VECTOR OPERATIONS TESTS ==========
    
    public void TestFill()
    {
        Console.WriteLine("\n=== Testing Fill Operation ===");
        int size = 100000;
        float value = 3.14f;

        using var v = lctx.GetVector(size);
        v.Fill(value);
        var result = v.ToHost();
        
        bool correct = result.All(x => Math.Abs(x - value) < EPSILON);
        Console.WriteLine($"Fill: {(correct ? "✓ PASS" : "✗ FAIL")}");
    }

    public void TestConcat()
    {
        Console.WriteLine("\n=== Testing Concat Operation ===");
        int size = 1000;
        var a = RandomVector(size);
        var b = RandomVector(size);

        using var ar = lctx.GetVector(size).Set(a);
        using var br = lctx.GetVector(size).Set(b);
        using var result = lctx.GetVector(size * 2);

        ar.Concat(br, result);
        var gpuResult = result.ToHost();

        var cpuResult = a.Concat(b).ToArray();
        bool correct = VectorsEqual(gpuResult, cpuResult);
        Console.WriteLine($"Concat: {(correct ? "✓ PASS" : "✗ FAIL")}");
    }

    public void TestSlice()
    {
        Console.WriteLine("\n=== Testing Slice Operation ===");
        int size = 1000;
        var a = RandomVector(size);
        int start = 100;
        int length = 200;

        using var ar = lctx.GetVector(size).Set(a);
        using var result = lctx.GetVector(length);

        ar.Slice(start, length, result);
        var gpuResult = result.ToHost();

        var cpuResult = a.Skip(start).Take(length).ToArray();
        bool correct = VectorsEqual(gpuResult, cpuResult);
        Console.WriteLine($"Slice: {(correct ? "✓ PASS" : "✗ FAIL")}");
    }

    public void TestNegate()
    {
        Console.WriteLine("\n=== Testing Negate ===");
        int size = 10000;
        var a = RandomVector(size, -100f, 100f);

        using var ar = lctx.GetVector(size).Set(a);
        using var result = lctx.GetVector(size);

        ar.Negate(result);
        var gpuResult = result.ToHost();
        var cpuResult = a.Select(x => -x).ToArray();
        bool correct = VectorsEqual(gpuResult, cpuResult);
        Console.WriteLine($"Negate: {(correct ? "✓ PASS" : "✗ FAIL")}");
    }

    public void TestOuterProduct()
    {
        Console.WriteLine("\n=== Testing Outer Product ===");
        int m = 50;
        int n = 50;
        var a = RandomVector(m);
        var b = RandomVector(n);

        using var ar = lctx.GetVector(m).Set(a);
        using var br = lctx.GetVector(n).Set(b);
        using var result = lctx.GetVector(m * n);

        ar.OuterProduct(br, n, result);
        var gpuResult = result.ToHost();

        var cpuResult = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                cpuResult[i * n + j] = a[i] * b[j];

        bool correct = VectorsEqual(gpuResult, cpuResult);
        Console.WriteLine($"OuterProduct: {(correct ? "✓ PASS" : "✗ FAIL")}");
    }

    // ========== PERFORMANCE BENCHMARKS ==========
    
    public void BenchmarkElementwiseOperations()
    {
        Console.WriteLine("\n=== Benchmark: Elementwise Operations ===");
        int[] sizes = { 1000, 10000, 100000, 1000000, 10000000 };

        foreach (int size in sizes)
        {
            var a = RandomVector(size);
            var b = RandomVector(size);

            using var ar = lctx.GetVector(size).Set(a);
            using var br = lctx.GetVector(size).Set(b);
            using var cr = lctx.GetVector(size);

            // Warmup
            ar.Add(br, cr);
            lctx.Synchronize();

            // Benchmark Add
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < 10; i++)
            {
                ar.Add(br, cr);
            }
            lctx.Synchronize();
            sw.Stop();
            double gpuTime = sw.ElapsedMilliseconds / 10.0;

            sw.Restart();
            for (int i = 0; i < 10; i++)
            {
                var _ = a.Zip(b, (x, y) => x + y).ToArray();
            }
            sw.Stop();
            double cpuTime = sw.ElapsedMilliseconds / 10.0;

            Console.WriteLine($"Size {size:D8}: GPU {gpuTime:F3}ms, CPU {cpuTime:F3}ms, Speedup {cpuTime/gpuTime:F2}x");
        }
    }

    public void BenchmarkTranscendentalFunctions()
    {
        Console.WriteLine("\n=== Benchmark: Transcendental Functions ===");
        int[] sizes = { 1000, 10000, 100000, 1000000 };

        foreach (int size in sizes)
        {
            var a = RandomVector(size, 0.1f, 3.14f);

            using var ar = lctx.GetVector(size).Set(a);
            using var result = lctx.GetVector(size);

            // Warmup
            ar.Sqrt(result);
            lctx.Synchronize();

            // Benchmark Sqrt
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < 10; i++)
            {
                ar.Sqrt(result);
            }
            lctx.Synchronize();
            sw.Stop();
            double gpuTime = sw.ElapsedMilliseconds / 10.0;

            sw.Restart();
            for (int i = 0; i < 10; i++)
            {
                var _ = a.Select(x => (float)Math.Sqrt(x)).ToArray();
            }
            sw.Stop();
            double cpuTime = sw.ElapsedMilliseconds / 10.0;

            Console.WriteLine($"Sqrt Size {size:D8}: GPU {gpuTime:F3}ms, CPU {cpuTime:F3}ms, Speedup {cpuTime/gpuTime:F2}x");
        }
    }

    public void BenchmarkDifferentDataSizes()
    {
        Console.WriteLine($"\n=== Benchmark: Different Data Sizes on {lctx.AcceleratorName} ===");
        int[] sizes = { 100, 1000, 10000, 100000, 1000000, 10000000 };

        foreach (int size in sizes)
        {
            var a = RandomVector(size);
            var b = RandomVector(size);

            using var ar = lctx.GetVector(size).Set(a);
            using var br = lctx.GetVector(size).Set(b);
            using var cr = lctx.GetVector(size);

            var sw = Stopwatch.StartNew();
            ar.Multiply(br, cr);
            lctx.Synchronize();
            sw.Stop();

            double gbPerSec = (size * sizeof(float) * 3 * 1e-9) / (sw.ElapsedMilliseconds * 1e-3);
            Console.WriteLine($"Size {size:D8}: {sw.ElapsedMilliseconds:D4}ms, {gbPerSec:F2} GB/s");
        }
    }

    // ========== STRESS TESTS ==========
    
    public void StressTestLargeOperations()
    {
        Console.WriteLine("\n=== Stress Test: Large Operations ===");
        try
        {
            int size = (int)Math.Pow(2, 24); // 16 million floats
            Console.WriteLine($"Allocating vectors of size {size}...");

            var a = RandomVector(size);
            var b = RandomVector(size);

            using var ar = lctx.GetVector(size).Set(a);
            using var br = lctx.GetVector(size).Set(b);
            using var cr = lctx.GetVector(size);

            var sw = Stopwatch.StartNew();
            ar.Add(br, cr);
            lctx.Synchronize();
            sw.Stop();

            var result = cr.ToHost();
            Console.WriteLine($"Successfully processed {size} elements in {sw.ElapsedMilliseconds}ms - ✓ PASS");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Stress test failed: {ex.Message} - ✗ FAIL");
        }
    }

    public void StressTestMemoryPool()
    {
        Console.WriteLine("\n=== Stress Test: Memory Pool ===");
        try
        {
            int iterations = 100;
            int size = 100000;
            
            for (int i = 0; i < iterations; i++)
            {
                var a = RandomVector(size);
                using var ar = lctx.GetVector(size).Set(a);
                using var result = lctx.GetVector(size);
                ar.Sqrt(result);
                _ = result.ToHost();
                
                if ((i + 1) % 20 == 0)
                    Console.WriteLine($"  Completed {i + 1}/{iterations} iterations");
            }
            Console.WriteLine("Memory pool stress test - ✓ PASS");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Memory pool stress test failed: {ex.Message} - ✗ FAIL");
        }
    }

    // ========== QUICK VERIFICATION TEST ==========
    
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
}