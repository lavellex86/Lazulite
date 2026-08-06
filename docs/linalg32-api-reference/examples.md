# Examples

### Sample Operations

{% code overflow="wrap" %}
```csharp
var (a, b, M) = (
    lctx.GetVector(5).Set([4, 5, 6, 7, 3]).AsVector(), 
    lctx.GetVector(128).Set([3, 5, 2, 3, 4]).AsVector(), 
    lctx.GetMatrix(5, 5)); // get some vectors to mess with, and a matrix
lctx.Fill(M, 1); // fill the matrix with 1s

var mag = lctx.L2Norm(a).Get(); // take the l2 norm, retrieve the scalar result 
var normalizedA = lctx.DivideScalar(a, mag); // normalize

var alpha = 0.01f; // rank 1 update; this is M += alpha * (a x b)
lctx.OuterProduct(a, b, r: M, alpha: alpha, beta: 1);
```
{% endcode %}

### Feed Forward Layer Example

{% code overflow="wrap" %}
```csharp
var (inputSize, outputSize) = (10, 20);
var input = lctx.GetVector(inputSize);
var weights = lctx.GetMatrix(outputSize, inputSize);
var biases = lctx.GetVector(outputSize);
var output = lctx.GetVector(outputSize);

lctx.MatrixVectorMultiply(weights, input, r: output);
lctx.Add(output, biases, r: output);
```
{% endcode %}

{% hint style="info" %}
You can view the full script for this page [here](https://github.com/lavellex86/Lazulite/blob/main/Tests/DocsExample6/Program.cs).
{% endhint %}
