using Lavelle.Lazulite;
using Lavelle.Stats32;
using Lavelle.Linalg32;

var lctx = new LazuliteContext().EnableLinalg32();
var sctx = new StatsContext(lctx);

var vec1 = new RemoteVector[]
{
    lctx.GetVector(3).Set([1, 2, 3]).AsVector(),
    lctx.GetVector(3).Set([4, 5, 6]).AsVector(),
    lctx.GetVector(3).Set([7, 8, 9]).AsVector()
};

var vec2 = new RemoteVector[]
{
    lctx.GetVector(3).Set([9, 8, 7]).AsVector(),
    lctx.GetVector(3).Set([6, 5, 4]).AsVector(),
    lctx.GetVector(3).Set([3, 2, 1]).AsVector()
};

var vec3 = new RemoteVector[]
{
    lctx.GetVector(3).Set([1, 2, 3]).AsVector(),
    lctx.GetVector(3).Set([4, 5, 6]).AsVector(),
    lctx.GetVector(3).Set([7, 8, 9]).AsVector(),
    lctx.GetVector(4).Set([1, 5, 3, 7]).AsVector()
};

var vec4 = new RemoteVector[]
{
    lctx.GetVector(3).Set([1, 2, 3]).AsVector(),
    lctx.GetVector(3).Set([4, 5, 6]).AsVector(),
    lctx.GetVector(3).Set([7, 8, 9]).AsVector(),
    lctx.GetVector(4).Set([11, 9, 2, 3]).AsVector()
};

var matrix1 = lctx.GetMatrix(4, 4)
                .Set(new float[,]
                        {
                            { 1, 2, 3, 4 },
                            { 22, 35, 104, 105 },
                            { 1029, 2338, 2127, 2130 },
                            { 2300, 6500, 10000, 9999 }
                        }
                    ).AsMatrix();

var matrix2 = lctx.GetMatrix(4, 4)
                .Set(new float[,]
                        {
                            { 4, 1, 3, 2 },
                            { 22, 104, 50, 500 },
                            { 10000, 3400, 750, 5000 },
                            { 14500, 9000, 3500, 7500 }
                        }
                    ).AsMatrix();

// Generate a 100x100 temporary matrix with random sample data
int size = 100;
float[,] tmp_mat100 = new float[size, size];
var rand = new Random(42);
for (int i = 0; i < size; i++)
{
    for (int j = 0; j < size; j++)
    {
        tmp_mat100[i, j] = (float)rand.NextDouble() * 100f;
    }
}
// Initiate a 100x100 RemoteMatrix and set the data in GPU memory
var matrix100 = lctx.GetMatrix(size, size).Set(tmp_mat100).AsMatrix();

lctx.Synchronize();
using var pearsonResults = sctx.Pearson(vec1[0], vec2[0]);
using var pearsonResults2 = sctx.Pearson(vec1[1], vec2[1]);
using var pearsonResults3 = sctx.Pearson(vec1[2], vec2[2]);
using var pearsonResults4 = sctx.Pearson(vec3[0], vec4[0]);
using var pearsonResults5 = sctx.Pearson(vec3[3], vec4[3]);

using var spearmanResults = sctx.Spearman(vec3[3], vec4[3]);


float pearsonValue = pearsonResults.Get();
float pearsonValue2 = pearsonResults2.Get();
float pearsonValue3 = pearsonResults3.Get();
float pearsonValue4 = pearsonResults4.Get();
float pearsonValue5 = pearsonResults5.Get();

float spearmanValue = spearmanResults.Get();


Console.WriteLine($"Pearson correlation between vec1[0] and vec2[0]: {pearsonValue}");
Console.WriteLine($"Pearson correlation between vec1[1] and vec2[1]: {pearsonValue2}");
Console.WriteLine($"Pearson correlation between vec1[2] and vec2[2]: {pearsonValue3}");
Console.WriteLine($"Pearson correlation between vec3[0] and vec4[0]: {pearsonValue4}");
Console.WriteLine($"Pearson correlation between vec3[3] and vec4[3]: {pearsonValue5}");
Console.WriteLine($"Spearman correlation between vec3[3] and vec4[3]: {spearmanValue}");

float[,] data = new float[matrix1.Shape[0], matrix1.Shape[1]];
data = sctx.PearsonMatrix(matrix1).Get();
int rows = data.GetLength(0);
int cols = data.GetLength(1);

Console.WriteLine("--- Pearson Correlation Matrix ---");
Console.Write("       ");
for (int j = 0; j < cols; j++)
{
    Console.Write($"V{j,-7}");
}

Console.WriteLine();
for (int i = 0; i < rows; i++)
{
    Console.Write($"V{i,-4} ");
    for (int j = 0; j < cols; j++)
    {
        Console.Write($"{data[i, j],7:F4} ");
    }
    Console.WriteLine();
}

float[,] data2 = new float[matrix2.Shape[0], matrix2.Shape[1]];
data2 = sctx.SpearmanMatrix(matrix2).Get();
int rows2 = data2.GetLength(0);
int cols2 = data2.GetLength(1);

Console.WriteLine("--- Spearman Correlation Matrix ---");
Console.Write("       ");
for (int j = 0; j < cols2; j++)
{
    Console.Write($"V{j,-7}");
}

Console.WriteLine();
for (int i = 0; i < rows2; i++)
{
    Console.Write($"V{i,-4} ");
    for (int j = 0; j < cols2; j++)
    {
        Console.Write($"{data2[i, j],7:F4} ");
    }
    Console.WriteLine();
}

Console.WriteLine("--- 100x100 Matrix Pearson Correlation ---");
var sw = System.Diagnostics.Stopwatch.StartNew();
float[,] pearsonResult100 = sctx.PearsonMatrix(matrix100).Get();
sw.Stop();
Console.WriteLine($"100x100 Pearson Correlation Matrix computed in: {sw.ElapsedMilliseconds} ms");
Console.WriteLine($"Result shape: [{pearsonResult100.GetLength(0)}, {pearsonResult100.GetLength(1)}]");
Console.WriteLine($"Sample Correlation [0, 1]: {pearsonResult100[0, 1]:F4}");
Console.WriteLine($"Sample Correlation [45, 50]: {pearsonResult100[45, 50]:F4}");
Console.WriteLine($"Sample Correlation [99, 98]: {pearsonResult100[99, 98]:F4}");

Console.WriteLine($"Running on: {lctx.Accelerator.Name}");
Console.WriteLine($"Device Type: {lctx.Accelerator.AcceleratorType}");

Console.WriteLine("--- 100x100 Matrix Spearman Correlation ---");
var sw1 = System.Diagnostics.Stopwatch.StartNew();
float[,] spearmanResult100 = sctx.SpearmanMatrix(matrix100).Get();
sw1.Stop();
Console.WriteLine($"100x100 Spearman Correlation Matrix computed in: {sw1.ElapsedMilliseconds} ms");
Console.WriteLine($"Result shape: [{spearmanResult100.GetLength(0)}, {spearmanResult100.GetLength(1)}]");
Console.WriteLine($"Sample Correlation [0, 1]: {spearmanResult100[0, 1]:F4}");
Console.WriteLine($"Sample Correlation [45, 50]: {spearmanResult100[45, 50]:F4}");
Console.WriteLine($"Sample Correlation [99, 98]: {spearmanResult100[99, 98]:F4}");

Console.WriteLine($"Running on: {lctx.Accelerator.Name}");
Console.WriteLine($"Device Type: {lctx.Accelerator.AcceleratorType}");