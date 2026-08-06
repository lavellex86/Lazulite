using Lavelle.Lazulite;
using Lavelle.Linalg32;

var lctx = new LazuliteContext();

var (a, b, M) = (
    lctx.GetVector(5).Set([4, 5, 6, 7, 3]).AsVector(), 
    lctx.GetVector(128).Set([3, 5, 2, 3, 4]).AsVector(), 
    lctx.GetMatrix(5, 5));
lctx.Fill(M, 1);

var mag = lctx.L2Norm(a).Get();
var normalizedA = lctx.DivideScalar(a, mag);

var alpha = 0.01f;
lctx.OuterProduct(a, b, r: M, alpha: alpha, beta: 1);

var (inputSize, outputSize) = (10, 20);
var input = lctx.GetVector(inputSize);
var weights = lctx.GetMatrix(outputSize, inputSize);
var biases = lctx.GetVector(outputSize);
var output = lctx.GetVector(outputSize);

lctx.MatrixVectorMultiply(weights, input, r: output);
lctx.Add(output, biases, r: output);
