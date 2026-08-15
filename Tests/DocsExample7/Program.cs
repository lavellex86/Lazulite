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