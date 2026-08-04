using Lavelle.Lazulite;
using Lavelle.Linalg32;

using var lctx = new LazuliteContext().EnableLinalg32();

var scalar = lctx.GetScalar(cleared: false); 
var vector = lctx.GetVector(5);
var matrix = lctx.GetMatrix(2, 2);

using var maybeScalar = (RemoteTensor<float>)scalar;
using var maybeVector = (RemoteTensor<float[]>)vector;
using var maybeMatrix = (RemoteTensor<float[,]>)matrix;

scalar = scalar.AsScalar();
vector = vector.AsVector();
matrix = matrix.AsMatrix();

using var a = lctx.GetVector(3);
using var b = lctx.GetVector(3);
using var sum = lctx.Add(a, b);
using var product = lctx.Multiply(a, b);

using var result1 = lctx.GetVector(3);
lctx.Divide(a, b, r: result1);
using var result2 = lctx.GetVector(3);
lctx.Subtract(a, b, result2);