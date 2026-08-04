# LinalgExtensions

{% hint style="info" %}
Try using `Ctrl-f` to find what you're looking for on this page.
{% endhint %}

```csharp
public static class LinalgExtensions
```

A set of linear algebra extensions for the `Lavelle.Lazulite` library.

***

### Tensor Extension Methods

```csharp
public static RemoteScalar AsScalar<T>(this RemoteTensor<T> tensor) where T : notnull
```

Interprets this tensor as a scalar.

***

```csharp
public static RemoteVector AsVector<T>(this RemoteTensor<T> tensor) where T : notnull
```

Interprets this tensor as a vector.

***

```csharp
public static RemoteMatrix AsMatrix<T>(this RemoteTensor<T> tensor) where T : notnull
```

Interprets this tensor as a matrix.

***

### Context Extension Methods

```csharp
public static RemoteScalar GetScalar(this LazuliteContext lctx, bool cleared = false)
```

Retrieves a scalar from the context pool.

***

```csharp
public static RemoteVector GetVector(this LazuliteContext lctx, int length, bool cleared = false)
```

Retrieves a vector from the context pool.

***

```csharp
public static RemoteMatrix GetMatrix(this LazuliteContext lctx, int m0, int m1, bool cleared = false)
```

Retrieves a matrix from the context pool.

***

```csharp
public static LazuliteContext EnableLinalg32(this LazuliteContext lctx)
```

Enables the `Linalg32` library on this context.

***

```csharp
public static RemoteTensor<T> Add<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull
```

Adds two tensors elementwise.

***

```csharp
public static RemoteTensor<T> Subtract<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull
```

Subtracts two tensors elementwise.

***

```csharp
public static RemoteTensor<T> Multiply<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull
```

Multiplies two tensors elementwise.

***

```csharp
public static RemoteTensor<T> Divide<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull
```

Divides two tensors elementwise.

***

```csharp
public static RemoteTensor<T> AddScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull
```

Adds a scalar value to every element of a tensor.

***

```csharp
public static RemoteTensor<T> SubtractScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull
```

Subtracts a scalar value from every element of a tensor.

***

```csharp
public static RemoteTensor<T> MultiplyScalar<T>(this LazuliteContext lctx, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null, bool useCuBlas = true) where T : notnull
```

Multiplies every element of a tensor by a scalar value. If `useCuBlas` is enabled and `r` is `a`, the CuBLAS version will be used.

***

```csharp
public static RemoteTensor<T> DivideScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull
```

Divides every element of a tensor by a scalar value.

***

```csharp
public static RemoteTensor<T> Negate<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Negates every element of a tensor.

***

```csharp
public static RemoteTensor<T> Reciprocal<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Takes the reciprocal of each element.

***

```csharp
public static RemoteTensor<T> Abs<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise absolute value of a tensor.

***

```csharp
public static RemoteTensor<T> Sign<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Returns the elementwise sign of a tensor (-1, 0, or 1 for each element).

***

```csharp
public static RemoteTensor<T> Min<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise minimum of two tensors.

***

```csharp
public static RemoteTensor<T> Max<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise maximum of two tensors.

***

```csharp
public static RemoteTensor<T> MinScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull
```

Clamps each element of a tensor to a maximum of `scalar`, returning the elementwise minimum.

***

```csharp
public static RemoteTensor<T> MaxScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull
```

Clamps each element of a tensor to a minimum of `scalar`, returning the elementwise maximum.

***

```csharp
public static RemoteTensor<T> Clamp<T>(this LazuliteContext _, RemoteTensor<T> a, float min, float max, RemoteTensor<T>? r = null) where T : notnull
```

Clamps the tensor elementwise between `min` and `max`.

***

```csharp
public static RemoteTensor<T> Fill<T>(this LazuliteContext lctx, RemoteTensor<T> tensor, float value) where T : notnull
```

Fills every element of a tensor in-place with the given scalar value and returns it.

***

```csharp
public static RemoteTensor<T> Axpy<T>(this LazuliteContext lctx, RemoteTensor<T> x, RemoteTensor<T> y, float alpha, RemoteTensor<T>? r = null, bool useCuBlas = true) where T : notnull
```

`r = alpha * x + y`. If `useCuBlas` is enabled, the result is `y += alpha * x`, so non-null `r` defaults to the non-CuBLAS version.

***

```csharp
public static RemoteTensor<T> Exp<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise natural exponential (e^x) of a tensor.

***

```csharp
public static RemoteTensor<T> Log<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise natural logarithm of a tensor.

***

```csharp
public static RemoteTensor<T> Log10<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise base-10 logarithm of a tensor.

***

```csharp
public static RemoteTensor<T> Log2<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise base-2 logarithm of a tensor.

***

```csharp
public static RemoteTensor<T> Sqrt<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise square root of a tensor.

***

```csharp
public static RemoteTensor<T> Pow<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull
```

Raises each element of `a` to the power of the corresponding element of `b`.

***

```csharp
public static RemoteTensor<T> PowScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull
```

Raises every element of a tensor to a scalar power.

***

```csharp
public static RemoteTensor<T> Sin<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise sine of a tensor.

***

```csharp
public static RemoteTensor<T> Cos<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise cosine of a tensor.

***

```csharp
public static RemoteTensor<T> Tan<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise tangent of a tensor.

***

```csharp
public static RemoteTensor<T> Sinh<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise hyperbolic sine of a tensor.

***

```csharp
public static RemoteTensor<T> Cosh<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise hyperbolic cosine of a tensor.

***

```csharp
public static RemoteTensor<T> Tanh<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise hyperbolic tangent of a tensor.

***

```csharp
public static RemoteTensor<T> Atan2<T>(this LazuliteContext _, RemoteTensor<T> y, RemoteTensor<T> x, RemoteTensor<T>? r = null) where T : notnull
```

Takes the 2-argument arctangent of `y` and `x` elementwise.

***

```csharp
public static RemoteTensor<T> Floor<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise floor of a tensor, rounding each element down to the nearest integer.

***

```csharp
public static RemoteTensor<T> Ceiling<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Computes the elementwise ceiling of a tensor, rounding each element up to the nearest integer.

***

```csharp
public static RemoteTensor<T> Round<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Rounds each element of a tensor to the nearest integer.

***

```csharp
public static RemoteTensor<T> Truncate<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull
```

Truncates each element of a tensor toward zero, discarding any fractional part.

***

```csharp
public static RemoteTensor<float[,]> OuterProduct(this LazuliteContext ctx, RemoteTensor<float[]> a, RemoteTensor<float[]> b, RemoteTensor<float[,]>? r = null, float alpha = 1f, float beta = 0f, bool useCuBlas = true)
```

Computes the outer product of two vectors, producing a matrix of shape `[a.Length, b.Length]` multiplied by `alpha` and accumulating into `r` scaled by `beta`.

***

```csharp
public static RemoteTensor<float[,]> MatrixMultiply(this LazuliteContext ctx, RemoteTensor<float[,]> a, RemoteTensor<float[,]> b, float alpha = 1.0f, float beta = 0.0f, RemoteTensor<float[,]>? r = null, bool transposeA = false, bool transposeB = false, bool useCuBlas = true)
```

Multiplies two matrices, optionally transposing either input. Supports scaling the product by `alpha` and accumulating into `r` scaled by `beta`.

***

```csharp
public static RemoteTensor<float[]> MatrixVectorMultiply(this LazuliteContext ctx, RemoteTensor<float[,]> m, RemoteTensor<float[]> v, float alpha = 1.0f, float beta = 0.0f, RemoteTensor<float[]>? r = null, bool transposeM = false, bool useCuBlas = true)
```

Multiplies a matrix by a vector, optionally transposing the matrix. Supports scaling via `alpha` and `beta` in the same manner as `MatrixMultiply`.

***

```csharp
public static RemoteTensor<float[,]> Transpose(this LazuliteContext ctx, RemoteTensor<float[,]> source, RemoteTensor<float[,]>? r = null)
```

Transposes a matrix.

***

```csharp
public static RemoteTensor<float[,]> BroadcastMatrixVectorAdd(this LazuliteContext _, RemoteTensor<float[,]> m, RemoteTensor<float[]> v, RemoteTensor<float[,]>? r = null)
```

Adds a vector to every row of a matrix, broadcasting the vector across the matrix's first dimension.

***

```csharp
public static RemoteTensor<float[]> NarrowcastVectorMatrixAdd(this LazuliteContext _, RemoteTensor<float[,]> m, RemoteTensor<float[]> r)
```

Reduces a matrix into a vector by summing each column across rows, then adds the result to `r`.

***

```csharp
public static RemoteVector Concat<T>(this LazuliteContext lctx, RemoteVector a, RemoteVector b, RemoteVector? r = null) where T : notnull
```

Concatenates two vectors.

***

```csharp
public static RemoteVector Slice<T>(this LazuliteContext lctx, RemoteVector source, int start, int end, RemoteVector? r = null) where T : notnull
```

Extracts a contiguous slice of a vector starting at the given index.

***

```csharp
public static RemoteTensor<float> Dot(this LazuliteContext lctx, RemoteTensor<float[]> a, RemoteTensor<float[]> b, RemoteTensor<float>? r = null, bool useCuBlas = true)
```

Takes the dot product of two vectors.

***

```csharp
public static RemoteTensor<float> L1Norm(this LazuliteContext lctx, RemoteTensor<float[]> v, RemoteTensor<float>? r = null, bool useCuBlas = true)
```

Takes the L1 norm of a vector (sum of absolutes).

***

```csharp
public static RemoteTensor<float> L2Norm(this LazuliteContext lctx, RemoteTensor<float[]> v, RemoteTensor<float>? r = null, bool useCuBlas = true)
```

Takes the L2 norm of a vector (sum of squares).

***

```csharp
public static RemoteTensor<float> Sum(this LazuliteContext lctx, RemoteTensor<float[]> v, RemoteTensor<float>? r = null)
```

Takes the sum of a vector.

***

```csharp
public static RemoteTensor<float[,]> CpuInvert(this LazuliteContext lctx, RemoteTensor<float[,]> matrix)
```

Inverts a square matrix on the CPU (syncing and transferring it from the compute device).

***

```csharp
public static float CpuDet(this LazuliteContext lctx, RemoteTensor<float[,]> matrix)
```

Computes the determinant of a square matrix on the CPU (syncing and transferring it from the compute device).

***

```csharp
public static float CpuL1Norm(this LazuliteContext lctx, RemoteTensor<float[]> vector)
```

Takes the L1 norm of a vector (sum of absolutes) on the CPU (syncing and transferring it from the compute device).

***

```csharp
public static float CpuL2Norm(this LazuliteContext lctx, RemoteTensor<float[]> vector)
```

Takes the L2 norm of a vector (sum of squares) on the CPU (syncing and transferring it from the compute device).

***

```csharp
public static float CpuSum(this LazuliteContext lctx, RemoteTensor<float[]> vector)
```

Takes the sum of a vector on the CPU (syncing and transferring it from the compute device).

***

```csharp
public static float CpuTrace(this LazuliteContext lctx, RemoteTensor<float[,]> matrix)
```

Takes the trace of a matrix on the CPU (syncing and transferring it from the compute device).

***

```csharp
public static int CpuArgMin(this LazuliteContext lctx, RemoteTensor<float[]> vector)
```

Finds the index of the minimum element on the CPU (syncing and transferring it from the compute device).

***

```csharp
public static int CpuArgMax(this LazuliteContext lctx, RemoteTensor<float[]> vector)
```

Finds the index of the maximum element on the CPU (syncing and transferring it from the compute device).

***

```csharp
public static (RemoteMatrix lu, RemoteVector piv, int sign) CpuDecomposeLU(this LazuliteContext lctx, RemoteMatrix x)
```

Takes the LU decomposition of a matrix on the CPU (syncing and transferring it from the compute device).

***

```csharp
public static (RemoteMatrix q, RemoteMatrix r) CpuDecomposeQR(this LazuliteContext lctx, RemoteMatrix x)
```

Takes the QR decomposition of a matrix on the CPU (syncing and transferring it from the compute device).

***

```csharp
public static RemoteVector LeastSquares(this LazuliteContext lctx, RemoteMatrix a, RemoteVector b)
```
Solves ||Ax - b||^2 for x.