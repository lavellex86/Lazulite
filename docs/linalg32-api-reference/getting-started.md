# Getting Started

To enable `Lavelle.Linalg32`, simply call the extension `.EnableLinalg32` on your `LazuliteContext`:

{% code overflow="wrap" %}
```csharp
var lctx = new LazuliteContext().EnableLinalg32();
```
{% endcode %}

Once you've done that, you're free to call any `Linalg32` extension from the context. To get a new scalar, vector, or matrix:

{% code overflow="wrap" %}
```csharp
var scalar = lctx.GetScalar(cleared: false); // grabs a new non-zero initialized scalar from the pool
var vector = lctx.GetVector(5); // grabs a length 5 vector from the pool; not cleared by default
var matrix = lctx.GetMatrix(2, 2); // grabs a 2x2 matrix from the pool
```
{% endcode %}

To interpret or cast a tensor to a scalar, vector, or matrix:

{% code overflow="wrap" %}
```csharp
var maybeScalar = (RemoteTensor<float>)scalar; // losing type info
var maybeVector = (RemoteTensor<float[]>)vector;
var maybeMatrix = (RemoteTensor<float[,]>)matrix;

scalar = scalar.AsScalar(); // casts back to RemoteScalar
vector = vector.AsVector(); // casts back to RemoteVector
matrix = matrix.AsMatrix(); // casts back to RemoteMatrix
// scalar = vector.AsScalar(); will throw, vector is not length 1
// vector = matrix.AsVector(); is fine, any tensor can become a vector becaus they're 1D buffer's underneath
```
{% endcode %}

Calling operations is simple:

{% code overflow="wrap" %}
```csharp
var a = lctx.GetVector(3);
var b = lctx.GetVector(3); // get some vectors
var sum = lctx.Add(a, b);
var product = lctx.Multiply(a, b); // add and multiply

using var result1 = lctx.GetVector(3);
lctx.Divide(a, b, r: result1); // set the result vector
using var result2 = lctx.GetVector(3);
lctx.Subtract(a, b, result2);
```
{% endcode %}

The full list of all operations in `Linalg32`:
- `Add`, `Subtract`, `Multiply`, `Divide`
- `AddScalar`, `SubtractScalar`, `MultiplyScalar`, `DivideScalar`
- `Negate`, `Reciprocal`, `Abs`, `Sign`
- `Min`, `Max`, `MinScalar`, `MaxScalar`
- `Clamp`, `Fill`, `Axpy`
- `Exp`, `Log`, `Log10`, `Log2`
- `Sqrt`, `Pow`, `PowScalar`, `Sin`
- `Cos`, `Tan`, `Sinh`, `Cosh`
- `Tanh`, `Atan2`, `Floor`, `Ceiling`
- `Round`, `Truncate`
- `OuterProduct`, `MatrixMultiply`, `MatrixVectorMultiply`, `Transpose`
- `BroadcastMatrixVectorAdd`, `NarrowcastVectorMatrixAdd`, `Concat`, `Slice`
- `Dot`, `L1Norm`, `L2Norm`, `Sum`
- `CpuInvert`, `CpuDet`, `CpuL1Norm`, `CpuL2Norm`
- `CpuSum`, `CpuTrace`, `CpuArgMin`, `CpuArgMax`
- `CpuDecomposeLU`, `CpuDecomposeQR`

Reference [linalgextensions.md](linalgextensions.md "mention") for details on all available operations.

Some operations use `alpha` and `beta` values; these use the form `r = f * alpha + r * beta`, where `f` is the operation. For example, in a matrix multiply, setting `beta = 1` allows us to accumulate the product into `r` rather than overwriting it.

Some operations also have a `useCuBlas` parameter, which will significantly speed up the operation. It's recommended to leave this on whenever possible (but note that it will conly take effect on CUDA compute devices).



{% hint style="info" %}
You can view the full script for this page [here](https://github.com/lavellex86/Lazulite/blob/main/Tests/DocsExample4/Program.cs).
{% endhint %}
