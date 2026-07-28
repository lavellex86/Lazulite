using System.Diagnostics;
using Raphael.Lazulite;
using Raphael.Linalg32;

// ── Config ────────────────────────────────────────────────────────────────────

const int WarmupRuns = 5;
const int BenchRuns = 20;

int[] matSizes = [64, 256, 512, 1024];
int[] vecSizes = [1_000, 100_000, 1_000_000, 10_000_000];
int[] poolSizes = [1_000, 100_000, 1_000_000, 10_000_000];

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
    Console.WriteLine($"│  {"Size",-20} {"Median (ms)",12} {"Throughput",18}");
    Console.WriteLine($"│  {"────────────────────",-20} {"────────────",12} {"──────────────────",18}");
}

void PrintRow(string label, double ms, string throughput) =>
    Console.WriteLine($"│  {label,-20} {ms,12:F3} {throughput,18}");

void PrintMicroRow(string label, double ms) =>
    Console.WriteLine($"│  {label,-20} {ms * 1000,12:F2} {"µs",18}");

void PrintFooter() => Console.WriteLine();

string VecLabel(int n) => n >= 1_000_000 ? $"{n / 1_000_000}M" : n >= 1_000 ? $"{n / 1_000}K" : $"{n}";

// ── 1. Matrix Multiply ────────────────────────────────────────────────────────

PrintHeader("Matrix Multiply  (A @ B, square NxN)");

foreach (int n in matSizes)
{
    var a = lctx.GetMatrix(n, n);
    var b = lctx.GetMatrix(n, n);
    var r = lctx.GetMatrix(n, n);

    for (int w = 0; w < WarmupRuns; w++) { lctx.MatrixMultiply(a, b, r: r); lctx.Synchronize(); }

    double ms = MeasureKernel(BenchRuns, () => lctx.MatrixMultiply(a, b, r: r));
    double gflops = 2.0 * n * n * n / (ms / 1000.0) / 1e9;
    PrintRow($"{n}x{n}", ms, $"{gflops:F2} GFLOP/s");

    a.Dispose(); b.Dispose(); r.Dispose();
}

PrintFooter();

// ── 2. Element-wise Ops ───────────────────────────────────────────────────────

var elementwiseOps = new (string Name, bool Binary, Func<RemoteVector, RemoteVector, RemoteVector, Action> Build)[]
{
    ("Add",      true,  (r, a, b) => () => lctx.Add(a, b, r)),
    ("Multiply", true,  (r, a, b) => () => lctx.Multiply(a, b, r)),
    ("Exp",      false, (r, a, b) => () => lctx.Exp(a, r)),
    ("Tanh",     false, (r, a, b) => () => lctx.Tanh(a, r)),
    ("Sqrt",     false, (r, a, b) => () => lctx.Sqrt(a, r)),
};

foreach (var (opName, binary, build) in elementwiseOps)
{
    PrintHeader($"Element-wise: {opName}");

    foreach (int len in vecSizes)
    {
        var a = lctx.GetVector(len);
        var b = lctx.GetVector(len);
        var r = lctx.GetVector(len);

        var dispatch = build(r, a, b);

        for (int w = 0; w < WarmupRuns; w++) { dispatch(); lctx.Synchronize(); }

        double ms = MeasureKernel(BenchRuns, dispatch);
        double gb = (double)len * sizeof(float) * (binary ? 3 : 2) / 1e9;
        double gbs = gb / (ms / 1000.0);
        PrintRow(VecLabel(len), ms, $"{gbs:F2} GB/s");

        a.Dispose(); b.Dispose(); r.Dispose();
    }

    PrintFooter();
}

// ── 3. Buffer Pool ────────────────────────────────────────────────────────────

PrintHeader("BufferPool: GetVector + Dispose round-trip");

foreach (int size in poolSizes)
{
    // Prime the pool so we measure the Stack path, not raw GPU allocation.
    lctx.GetVector(size).Dispose();

    for (int w = 0; w < WarmupRuns; w++) lctx.GetVector(size).Dispose();

    double ms = MeasureCpu(BenchRuns, () => lctx.GetVector(size).Dispose());
    PrintMicroRow($"{VecLabel(size)} floats", ms);
}

PrintFooter();
Console.WriteLine("Done.");