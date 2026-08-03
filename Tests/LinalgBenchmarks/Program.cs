using System.Diagnostics;
using Lavelle.Lazulite;
using Lavelle.Linalg32;

// ── Config ────────────────────────────────────────────────────────────────────

const int WarmupRuns = 5;
const int BenchRuns = 20;

int[] matSizes = [64, 128, 256, 512, 1024, 2048];
int[] vecSizes = [1_000, 100_000, 1_000_000, 10_000_000];
int[] poolSizes = [1_000, 100_000, 1_000_000, 10_000_000];
int[] mvSizes = [64, 256, 512, 1024, 2048];

// ── Setup ─────────────────────────────────────────────────────────────────────

using var lctx = new LazuliteContext(gpu: true);
lctx.EnableLinalg32();

Console.WriteLine($"Accelerator : {lctx.AcceleratorName}");
Console.WriteLine();

var sw = new Stopwatch();

// ── Helpers ───────────────────────────────────────────────────────────────────

double MeasureKernel(int runs, Action dispatch)
{
    var times = new double[runs];
    for (int r = 0; r < runs; r++)
    {
        sw.Restart();
        dispatch();
        lctx.Synchronize();
        sw.Stop();
        times[r] = sw.Elapsed.TotalMilliseconds;
    }
    Array.Sort(times);
    return times[runs / 2];
}

double MeasureCpu(int runs, Action work)
{
    var times = new double[runs];
    for (int r = 0; r < runs; r++)
    {
        sw.Restart();
        work();
        sw.Stop();
        times[r] = sw.Elapsed.TotalMilliseconds;
    }
    Array.Sort(times);
    return times[runs / 2];
}

void PrintHeader(string title)
{
    Console.WriteLine($"┌─ {title}");
    Console.WriteLine($"│  {"Size",-24} {"Median (ms)",12} {"Throughput",20}");
    Console.WriteLine($"│  {"────────────────────────",-24} {"────────────",12} {"────────────────────",20}");
}

void PrintRow(string label, double ms, string throughput) =>
    Console.WriteLine($"│  {label,-24} {ms,12:F3} {throughput,20}");

void PrintMicroRow(string label, double ms) =>
    Console.WriteLine($"│  {label,-24} {ms * 1000.0,12:F2} {"µs",20}");

void PrintNanoRow(string label, double ms) =>
    Console.WriteLine($"│  {label,-24} {ms * 1_000_000.0,12:F1} {"ns",20}");

void PrintSeparator(string label) =>
    Console.WriteLine($"│  ── {label}");

void PrintFooter() => Console.WriteLine();

string VecLabel(int n) => n >= 1_000_000 ? $"{n / 1_000_000}M" : n >= 1_000 ? $"{n / 1_000}K" : $"{n}";
string MatLabel(int n) => $"{n}x{n}";

// ── 1. Matrix Multiply: cuBLAS vs Naive ──────────────────────────────────────

PrintHeader("Matrix Multiply: cuBLAS vs Naive  (A @ B, square NxN)");

foreach (int n in matSizes)
{
    var a = lctx.GetMatrix(n, n);
    var b = lctx.GetMatrix(n, n);
    var r = lctx.GetMatrix(n, n);

    // cuBLAS
    for (int w = 0; w < WarmupRuns; w++) { lctx.MatrixMultiply(a, b, r: r, useCuBlas: true); lctx.Synchronize(); }
    double msCublas = MeasureKernel(BenchRuns, () => lctx.MatrixMultiply(a, b, r: r, useCuBlas: true));
    double gflopsCublas = 2.0 * n * n * n / (msCublas / 1000.0) / 1e9;
    PrintRow($"{MatLabel(n)} cuBLAS", msCublas, $"{gflopsCublas:F2} GFLOP/s");

    // Naive kernel (only up to 512 — larger gets very slow)
    if (n <= 512)
    {
        for (int w = 0; w < WarmupRuns; w++) { lctx.MatrixMultiply(a, b, r: r, useCuBlas: false); lctx.Synchronize(); }
        double msNaive = MeasureKernel(BenchRuns, () => lctx.MatrixMultiply(a, b, r: r, useCuBlas: false));
        double gflopsNaive = 2.0 * n * n * n / (msNaive / 1000.0) / 1e9;
        PrintRow($"{MatLabel(n)} naive", msNaive, $"{gflopsNaive:F2} GFLOP/s");
    }

    a.Dispose(); b.Dispose(); r.Dispose();
}

PrintFooter();

// ── 2. Matrix-Vector Multiply: cuBLAS vs Naive ───────────────────────────────

PrintHeader("Matrix-Vector Multiply (Ax): cuBLAS vs Naive");

foreach (int n in mvSizes)
{
    var m = lctx.GetMatrix(n, n);
    var v = lctx.GetVector(n);
    var r = lctx.GetVector(n);

    for (int w = 0; w < WarmupRuns; w++) { lctx.MatrixVectorMultiply(m, v, n, r: r, useCuBlas: true); lctx.Synchronize(); }
    double msCublas = MeasureKernel(BenchRuns, () => lctx.MatrixVectorMultiply(m, v, n, r: r, useCuBlas: true));
    double gflopsCublas = 2.0 * n * n / (msCublas / 1000.0) / 1e9;
    PrintRow($"{MatLabel(n)} cuBLAS", msCublas, $"{gflopsCublas:F3} GFLOP/s");

    for (int w = 0; w < WarmupRuns; w++) { lctx.MatrixVectorMultiply(m, v, n, r: r, useCuBlas: false); lctx.Synchronize(); }
    double msNaive = MeasureKernel(BenchRuns, () => lctx.MatrixVectorMultiply(m, v, n, r: r, useCuBlas: false));
    double gflopsNaive = 2.0 * n * n / (msNaive / 1000.0) / 1e9;
    PrintRow($"{MatLabel(n)} naive", msNaive, $"{gflopsNaive:F3} GFLOP/s");

    m.Dispose(); v.Dispose(); r.Dispose();
}

PrintFooter();

// ── 3. Matrix Multiply with Transpose Variants ───────────────────────────────

PrintHeader("Matrix Multiply Transpose Variants (512x512, cuBLAS)");
{
    int n = 512;
    var a = lctx.GetMatrix(n, n);
    var b = lctx.GetMatrix(n, n);
    var r = lctx.GetMatrix(n, n);

    var variants = new (string Label, bool tA, bool tB)[]
    {
        ("A @ B",    false, false),
        ("Aᵀ @ B",   true,  false),
        ("A @ Bᵀ",   false, true),
        ("Aᵀ @ Bᵀ",  true,  true),
    };

    foreach (var (label, tA, tB) in variants)
    {
        for (int w = 0; w < WarmupRuns; w++) { lctx.MatrixMultiply(a, b, r: r, transposeA: tA, transposeB: tB); lctx.Synchronize(); }
        double ms = MeasureKernel(BenchRuns, () => lctx.MatrixMultiply(a, b, r: r, transposeA: tA, transposeB: tB));
        double gflops = 2.0 * n * n * n / (ms / 1000.0) / 1e9;
        PrintRow(label, ms, $"{gflops:F2} GFLOP/s");
    }

    a.Dispose(); b.Dispose(); r.Dispose();
}
PrintFooter();

// ── 4. Dot Product: cuBLAS vs Naive ──────────────────────────────────────────

PrintHeader("Dot Product: cuBLAS vs Naive");

foreach (int len in vecSizes)
{
    var a = lctx.GetVector(len);
    var b = lctx.GetVector(len);
    var rs = lctx.GetScalar(true);

    for (int w = 0; w < WarmupRuns; w++) { lctx.Dot(a, b, rs, useCuBlas: true); lctx.Synchronize(); }
    double msCublas = MeasureKernel(BenchRuns, () => lctx.Dot(a, b, rs, useCuBlas: true));
    double gbCublas = 2.0 * len * sizeof(float) / 1e9;
    PrintRow($"{VecLabel(len)} cuBLAS", msCublas, $"{gbCublas / (msCublas / 1000.0):F2} GB/s");

    for (int w = 0; w < WarmupRuns; w++) { lctx.Dot(a, b, rs, useCuBlas: false); lctx.Synchronize(); }
    double msNaive = MeasureKernel(BenchRuns, () => lctx.Dot(a, b, rs, useCuBlas: false));
    PrintRow($"{VecLabel(len)} naive", msNaive, $"{gbCublas / (msNaive / 1000.0):F2} GB/s");

    a.Dispose(); b.Dispose(); rs.Dispose();
}

PrintFooter();

// ── 5. AXPY: cuBLAS vs Naive ─────────────────────────────────────────────────

PrintHeader("AXPY (y += α·x): cuBLAS vs Naive");

foreach (int len in vecSizes)
{
    var x = lctx.GetVector(len);
    var y = lctx.GetVector(len);

    for (int w = 0; w < WarmupRuns; w++) { lctx.Axpy(x, y, 2.0f, useCuBlas: true); lctx.Synchronize(); }
    double msCublas = MeasureKernel(BenchRuns, () => lctx.Axpy(x, y, 2.0f, useCuBlas: true));
    double gbAxpy = 3.0 * len * sizeof(float) / 1e9;
    PrintRow($"{VecLabel(len)} cuBLAS", msCublas, $"{gbAxpy / (msCublas / 1000.0):F2} GB/s");

    for (int w = 0; w < WarmupRuns; w++) { lctx.Axpy(x, y, 2.0f, r: y, useCuBlas: false); lctx.Synchronize(); }
    double msNaive = MeasureKernel(BenchRuns, () => lctx.Axpy(x, y, 2.0f, r: y, useCuBlas: false));
    PrintRow($"{VecLabel(len)} naive", msNaive, $"{gbAxpy / (msNaive / 1000.0):F2} GB/s");

    x.Dispose(); y.Dispose();
}

PrintFooter();

// ── 6. ScalarMultiply (Scal): cuBLAS vs Naive ────────────────────────────────

PrintHeader("Scalar Multiply (α·x): cuBLAS vs Naive");

foreach (int len in vecSizes)
{
    var a = lctx.GetVector(len);

    // cuBLAS scal: in-place (r == a)
    for (int w = 0; w < WarmupRuns; w++) { lctx.MultiplyScalar(a, 1.5f, r: a, useCuBlas: true); lctx.Synchronize(); }
    double msCublas = MeasureKernel(BenchRuns, () => lctx.MultiplyScalar(a, 1.5f, r: a, useCuBlas: true));
    double gbScal = 2.0 * len * sizeof(float) / 1e9;
    PrintRow($"{VecLabel(len)} cuBLAS", msCublas, $"{gbScal / (msCublas / 1000.0):F2} GB/s");

    // Naive kernel: r != a
    var rNaive = lctx.GetVector(len);
    for (int w = 0; w < WarmupRuns; w++) { lctx.MultiplyScalar(a, 1.5f, r: rNaive, useCuBlas: false); lctx.Synchronize(); }
    double msNaive = MeasureKernel(BenchRuns, () => lctx.MultiplyScalar(a, 1.5f, r: rNaive, useCuBlas: false));
    PrintRow($"{VecLabel(len)} naive", msNaive, $"{gbScal / (msNaive / 1000.0):F2} GB/s");

    a.Dispose(); rNaive.Dispose();
}

PrintFooter();

// ── 7. Outer Product: cuBLAS vs Naive ────────────────────────────────────────

PrintHeader("Outer Product (aᵀb): cuBLAS vs Naive");

foreach (int n in new int[] { 64, 256, 512, 1024})
{
    var a = lctx.GetVector(n);
    var b = lctx.GetVector(n);
    var r = lctx.GetMatrix(n, n);

    for (int w = 0; w < WarmupRuns; w++) { lctx.OuterProduct(a, b, r, useCuBlas: true); lctx.Synchronize(); }
    double msCublas = MeasureKernel(BenchRuns, () => lctx.OuterProduct(a, b, r, useCuBlas: true));
    double gbOuter = (double)(2 * n + n * n) * sizeof(float) / 1e9;
    PrintRow($"n={n} cuBLAS", msCublas, $"{gbOuter / (msCublas / 1000.0):F2} GB/s");

    for (int w = 0; w < WarmupRuns; w++) { lctx.OuterProduct(a, b, r, useCuBlas: false); lctx.Synchronize(); }
    double msNaive = MeasureKernel(BenchRuns, () => lctx.OuterProduct(a, b, r, useCuBlas: false));
    PrintRow($"n={n} naive", msNaive, $"{gbOuter / (msNaive / 1000.0):F2} GB/s");

    a.Dispose(); b.Dispose(); r.Dispose();
}

PrintFooter();

// ── 8. Transpose ─────────────────────────────────────────────────────────────

PrintHeader("Matrix Transpose  (NxN)");

foreach (int n in matSizes)
{
    var a = lctx.GetMatrix(n, n);
    var r = lctx.GetMatrix(n, n);

    for (int w = 0; w < WarmupRuns; w++) { lctx.Transpose(a, r); lctx.Synchronize(); }
    double ms = MeasureKernel(BenchRuns, () => lctx.Transpose(a, r));
    double gb = 2.0 * n * n * sizeof(float) / 1e9;
    PrintRow(MatLabel(n), ms, $"{gb / (ms / 1000.0):F2} GB/s");

    a.Dispose(); r.Dispose();
}

PrintFooter();

// ── 9. Broadcast Matrix-Vector Add ───────────────────────────────────────────

PrintHeader("Broadcast Matrix-Vector Add  (M rows += v)");

foreach (int n in matSizes)
{
    var m = lctx.GetMatrix(n, n);
    var v = lctx.GetVector(n);
    var r = lctx.GetMatrix(n, n);

    for (int w = 0; w < WarmupRuns; w++) { lctx.BroadcastMatrixVectorAdd(m, v, r); lctx.Synchronize(); }
    double ms = MeasureKernel(BenchRuns, () => lctx.BroadcastMatrixVectorAdd(m, v, r));
    double gb = (2.0 * n * n + n) * sizeof(float) / 1e9;
    PrintRow(MatLabel(n), ms, $"{gb / (ms / 1000.0):F2} GB/s");

    m.Dispose(); v.Dispose(); r.Dispose();
}

PrintFooter();

// ── 10. CPU Matrix Inversion ──────────────────────────────────────────────────

PrintHeader("CPU Matrix Inversion  (LU, square NxN)");

foreach (int n in new int[] { 16, 32, 64, 128, 256})
{
    // Allocate a diagonally dominant matrix so it's invertible
    var hostData = new float[n, n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++) hostData[i, j] = (i == j) ? n : 1f;
    }
    var remoteM = (RemoteMatrix)lctx.GetMatrix(n, n).Set(hostData);

    for (int w = 0; w < WarmupRuns; w++) { var inv = lctx.CpuInvert(remoteM); inv.Dispose(); }
    double ms = MeasureCpu(BenchRuns, () => { var inv = lctx.CpuInvert(remoteM); inv.Dispose(); });

    if (ms < 1.0)
        PrintMicroRow(MatLabel(n), ms);
    else
        PrintRow(MatLabel(n), ms, "");

    remoteM.Dispose();
}

PrintFooter();

// ── 11. Element-wise Ops (full suite) ────────────────────────────────────────

var elementwiseOps = new (string Name, bool Binary, Func<RemoteVector, RemoteVector, RemoteVector, Action> Build)[]
{
    ("Add",      true,  (r, a, b) => () => lctx.Add(a, b, r)),
    ("Subtract", true,  (r, a, b) => () => lctx.Subtract(a, b, r)),
    ("Multiply", true,  (r, a, b) => () => lctx.Multiply(a, b, r)),
    ("Divide",   true,  (r, a, b) => () => lctx.Divide(a, b, r)),
    ("Min",      true,  (r, a, b) => () => lctx.Min(a, b, r)),
    ("Max",      true,  (r, a, b) => () => lctx.Max(a, b, r)),
    ("Pow",      true,  (r, a, b) => () => lctx.Pow(a, b, r)),
    ("Exp",      false, (r, a, b) => () => lctx.Exp(a, r)),
    ("Log",      false, (r, a, b) => () => lctx.Log(a, r)),
    ("Log2",     false, (r, a, b) => () => lctx.Log2(a, r)),
    ("Log10",    false, (r, a, b) => () => lctx.Log10(a, r)),
    ("Sqrt",     false, (r, a, b) => () => lctx.Sqrt(a, r)),
    ("Sin",      false, (r, a, b) => () => lctx.Sin(a, r)),
    ("Cos",      false, (r, a, b) => () => lctx.Cos(a, r)),
    ("Tan",      false, (r, a, b) => () => lctx.Tan(a, r)),
    ("Sinh",     false, (r, a, b) => () => lctx.Sinh(a, r)),
    ("Cosh",     false, (r, a, b) => () => lctx.Cosh(a, r)),
    ("Tanh",     false, (r, a, b) => () => lctx.Tanh(a, r)),
    ("Abs",      false, (r, a, b) => () => lctx.Abs(a, r)),
    ("Negate",   false, (r, a, b) => () => lctx.Negate(a, r)),
    ("Sign",     false, (r, a, b) => () => lctx.Sign(a, r)),
    ("Floor",    false, (r, a, b) => () => lctx.Floor(a, r)),
    ("Ceiling",  false, (r, a, b) => () => lctx.Ceiling(a, r)),
    ("Round",    false, (r, a, b) => () => lctx.Round(a, r)),
    ("Truncate", false, (r, a, b) => () => lctx.Truncate(a, r)),
};

foreach (var (opName, binary, build) in elementwiseOps)
{
    PrintHeader($"Element-wise: {opName}  ({(binary ? "binary" : "unary")})");

    foreach (int len in vecSizes)
    {
        var a = lctx.GetVector(len);
        var b = lctx.GetVector(len);
        var r = lctx.GetVector(len);

        var dispatch = build(r, a, b);

        for (int w = 0; w < WarmupRuns; w++) { dispatch(); lctx.Synchronize(); }

        double ms = MeasureKernel(BenchRuns, dispatch);
        double gb = (double)len * sizeof(float) * (binary ? 3 : 2) / 1e9;
        PrintRow(VecLabel(len), ms, $"{gb / (ms / 1000.0):F2} GB/s");

        a.Dispose(); b.Dispose(); r.Dispose();
    }

    PrintFooter();
}

// ── 12. Scalar Operations (AddScalar, SubtractScalar, etc.) ──────────────────

var scalarOps = new (string Name, Func<RemoteVector, RemoteVector, Action> Build)[]
{
    ("AddScalar",      (r, a) => () => lctx.AddScalar(a, 1f, r)),
    ("SubtractScalar", (r, a) => () => lctx.SubtractScalar(a, 1f, r)),
    ("MultiplyScalar", (r, a) => () => lctx.MultiplyScalar(a, 2f, r, useCuBlas: false)),
    ("DivideScalar",   (r, a) => () => lctx.DivideScalar(a, 2f, r)),
    ("MinScalar",      (r, a) => () => lctx.MinScalar(a, 0f, r)),
    ("MaxScalar",      (r, a) => () => lctx.MaxScalar(a, 0f, r)),
    ("PowScalar",      (r, a) => () => lctx.PowScalar(a, 2f, r)),
    ("Fill",           (r, a) => () => lctx.Fill(a, 3.14f)),
};

PrintHeader("Scalar-broadcast ops  (1 read + 1 write)");
// Show at 10M only to keep output concise
{
    int len = 10_000_000;
    foreach (var (opName, build) in scalarOps)
    {
        var a = lctx.GetVector(len);
        var r = lctx.GetVector(len);
        var dispatch = build(r, a);
        for (int w = 0; w < WarmupRuns; w++) { dispatch(); lctx.Synchronize(); }
        double ms = MeasureKernel(BenchRuns, dispatch);
        double gb = 2.0 * len * sizeof(float) / 1e9;
        PrintRow(opName, ms, $"{gb / (ms / 1000.0):F2} GB/s");
        a.Dispose(); r.Dispose();
    }
}
PrintFooter();

// ── 13. Concat & Slice ────────────────────────────────────────────────────────

PrintHeader("Concat & Slice  (vector ops)");

foreach (int len in vecSizes)
{
    var a = lctx.GetVector(len);
    var b = lctx.GetVector(len);

    // Concat
    {
        var rCat = lctx.GetVector(len * 2);
        for (int w = 0; w < WarmupRuns; w++) { lctx.Concat<float[]>(a, b, rCat); lctx.Synchronize(); }
        double ms = MeasureKernel(BenchRuns, () => lctx.Concat<float[]>(a, b, rCat));
        double gb = 3.0 * len * sizeof(float) / 1e9;
        PrintRow($"{VecLabel(len)} concat", ms, $"{gb / (ms / 1000.0):F2} GB/s");
        rCat.Dispose();
    }

    // Slice (first half)
    {
        var src = lctx.GetVector(len * 2);
        var rSlice = lctx.GetVector(len);
        for (int w = 0; w < WarmupRuns; w++) { lctx.Slice<float[]>(src, 0, len, rSlice); lctx.Synchronize(); }
        double ms = MeasureKernel(BenchRuns, () => lctx.Slice<float[]>(src, 0, len, rSlice));
        double gb = 2.0 * len * sizeof(float) / 1e9;
        PrintRow($"{VecLabel(len)} slice", ms, $"{gb / (ms / 1000.0):F2} GB/s");
        src.Dispose(); rSlice.Dispose();
    }

    a.Dispose(); b.Dispose();
}

PrintFooter();

// ── 14. Host ↔ Device Transfer ────────────────────────────────────────────────

PrintHeader("Host ↔ Device Transfer (Get / Set)");

foreach (int len in vecSizes)
{
    var hostArr = new float[len];
    var remote = lctx.GetVector(len);

    // Upload (Set)
    for (int w = 0; w < WarmupRuns; w++) remote.Set(hostArr);
    double msUp = MeasureCpu(BenchRuns, () => remote.Set(hostArr));
    double gbUp = (double)len * sizeof(float) / 1e9;
    PrintRow($"{VecLabel(len)} upload", msUp, $"{gbUp / (msUp / 1000.0):F2} GB/s");

    // Download (Get)
    for (int w = 0; w < WarmupRuns; w++) remote.Get();
    double msDown = MeasureCpu(BenchRuns, () => remote.Get());
    PrintRow($"{VecLabel(len)} download", msDown, $"{gbUp / (msDown / 1000.0):F2} GB/s");

    remote.Dispose();
}

PrintFooter();

// ── 15. Buffer Pool ───────────────────────────────────────────────────────────

PrintHeader("BufferPool: GetVector + Dispose round-trip");

foreach (int size in poolSizes)
{
    lctx.GetVector(size).Dispose();
    for (int w = 0; w < WarmupRuns; w++) lctx.GetVector(size).Dispose();

    double ms = MeasureCpu(BenchRuns, () => lctx.GetVector(size).Dispose());
    PrintMicroRow($"{VecLabel(size)} floats", ms);
}

PrintFooter();

// ── 16. Kernel Launch Overhead ────────────────────────────────────────────────

PrintHeader("Kernel Launch Overhead  (tiny 1-element tensor)");
{
    var tiny = lctx.GetVector(1);
    var tinyB = lctx.GetVector(1);
    var tinyR = lctx.GetVector(1);

    var launchOps = new (string Name, Action Dispatch)[]
    {
        ("Add",     () => lctx.Add(tiny, tinyB, tinyR)),
        ("Exp",     () => lctx.Exp(tiny, tinyR)),
        ("Tanh",    () => lctx.Tanh(tiny, tinyR)),
        ("Fill",    () => lctx.Fill(tiny, 1f)),
    };

    foreach (var (name, dispatch) in launchOps)
    {
        for (int w = 0; w < WarmupRuns; w++) { dispatch(); lctx.Synchronize(); }
        double ms = MeasureKernel(BenchRuns, dispatch);
        PrintNanoRow(name, ms);
    }

    tiny.Dispose(); tinyB.Dispose(); tinyR.Dispose();
}

PrintFooter();

// ── 17. Fused Patterns (common NN building blocks) ───────────────────────────

PrintHeader("Fused Pattern: Linear (Wx + b)  — Gemv + BroadcastAdd");

foreach (int n in mvSizes)
{
    var W = lctx.GetMatrix(n, n);
    var x = lctx.GetVector(n);
    var b = lctx.GetVector(n);
    var y = lctx.GetVector(n);

    Action linear = () =>
    {
        lctx.MatrixVectorMultiply(W, x, n, r: y);
        // in-place add bias by reusing y as both source and result of Axpy
        lctx.Axpy(b, y, 1f, useCuBlas: true);
    };

    for (int w = 0; w < WarmupRuns; w++) { linear(); lctx.Synchronize(); }
    double ms = MeasureKernel(BenchRuns, linear);
    double gflops = (2.0 * n * n + n) / (ms / 1000.0) / 1e9;
    PrintRow(MatLabel(n), ms, $"{gflops:F3} GFLOP/s");

    W.Dispose(); x.Dispose(); b.Dispose(); y.Dispose();
}

PrintFooter();

PrintHeader("Fused Pattern: Sigmoid (1 / (1 + e^-x)) — Negate + Exp + AddScalar");

foreach (int len in vecSizes)
{
    var x = lctx.GetVector(len);
    var r = lctx.GetVector(len);

    Action sigmoid = () =>
    {
        lctx.Negate(x, r);
        lctx.Exp(r, r);
        lctx.AddScalar(r, 1f, r);
        // reciprocal via DivideScalar(1/x) isn't in the API, so we PowScalar -1
        lctx.PowScalar(r, -1f, r);
    };

    for (int w = 0; w < WarmupRuns; w++) { sigmoid(); lctx.Synchronize(); }
    double ms = MeasureKernel(BenchRuns, sigmoid);
    double gb = 2.0 * len * sizeof(float) / 1e9;
    PrintRow(VecLabel(len), ms, $"{gb / (ms / 1000.0):F2} GB/s");

    x.Dispose(); r.Dispose();
}

PrintFooter();

PrintHeader("Fused Pattern: ReLU (max(x, 0))  — MaxScalar");

foreach (int len in vecSizes)
{
    var x = lctx.GetVector(len);
    var r = lctx.GetVector(len);

    for (int w = 0; w < WarmupRuns; w++) { lctx.MaxScalar(x, 0f, r); lctx.Synchronize(); }
    double ms = MeasureKernel(BenchRuns, () => lctx.MaxScalar(x, 0f, r));
    double gb = 2.0 * len * sizeof(float) / 1e9;
    PrintRow(VecLabel(len), ms, $"{gb / (ms / 1000.0):F2} GB/s");

    x.Dispose(); r.Dispose();
}

PrintFooter();

// ── Done ──────────────────────────────────────────────────────────────────────
Console.WriteLine("Done.");