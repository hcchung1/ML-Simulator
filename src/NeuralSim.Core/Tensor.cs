namespace NeuralSim.Core;

/// <summary>
/// N-dimensional tensor backed by a flat float array.
/// MLP uses 2D (batch, features); CNN uses 4D (batch, channels, h, w);
/// Transformer uses 3D (batch, seq, dim).
/// </summary>
public sealed class Tensor
{
    public float[] Data { get; }
    public int[] Shape { get; }

    /// <summary>Total number of elements.</summary>
    public int Length => Data.Length;

    /// <summary>Number of dimensions.</summary>
    public int Rank => Shape.Length;

    public Tensor(int[] shape)
    {
        Shape = shape;
        Data = new float[ShapeToLength(shape)];
    }

    public Tensor(int[] shape, float[] data)
    {
        if (data.Length != ShapeToLength(shape))
            throw new ArgumentException(
                $"Data length {data.Length} doesn't match shape [{string.Join(',', shape)}] = {ShapeToLength(shape)}");
        Shape = shape;
        Data = data;
    }

    // ───── indexing helpers ─────

    /// <summary>Access element by flat index.</summary>
    public float this[int i]
    {
        get => Data[i];
        set => Data[i] = value;
    }

    /// <summary>2D accessor [row, col].</summary>
    public float Get2D(int r, int c) => Data[r * Shape[1] + c];
    public void Set2D(int r, int c, float v) => Data[r * Shape[1] + c] = v;

    /// <summary>3D accessor [d0, d1, d2].</summary>
    public float Get3D(int d0, int d1, int d2) => Data[(d0 * Shape[1] + d1) * Shape[2] + d2];
    public void Set3D(int d0, int d1, int d2, float v) => Data[(d0 * Shape[1] + d1) * Shape[2] + d2] = v;

    /// <summary>4D accessor [d0, d1, d2, d3].</summary>
    public float Get4D(int d0, int d1, int d2, int d3) => Data[((d0 * Shape[1] + d1) * Shape[2] + d2) * Shape[3] + d3];
    public void Set4D(int d0, int d1, int d2, int d3, float v) => Data[((d0 * Shape[1] + d1) * Shape[2] + d2) * Shape[3] + d3] = v;

    // ───── factory helpers ─────

    public static Tensor Zeros(params int[] shape) => new(shape);

    public static Tensor FromArray1D(float[] arr) => new([arr.Length], arr);

    public static Tensor FromArray2D(float[,] arr)
    {
        int rows = arr.GetLength(0), cols = arr.GetLength(1);
        var flat = new float[rows * cols];
        Buffer.BlockCopy(arr, 0, flat, 0, flat.Length * sizeof(float));
        return new Tensor([rows, cols], flat);
    }

    /// <summary>Random uniform [0,1) tensor.</summary>
    public static Tensor Random(Random rng, params int[] shape)
    {
        var t = new Tensor(shape);
        for (int i = 0; i < t.Length; i++)
            t.Data[i] = (float)rng.NextDouble();
        return t;
    }

    /// <summary>Xavier / Glorot uniform init for weight matrices.</summary>
    public static Tensor XavierUniform(Random rng, int fanIn, int fanOut)
    {
        float limit = MathF.Sqrt(6f / (fanIn + fanOut));
        var t = new Tensor([fanIn, fanOut]);
        for (int i = 0; i < t.Length; i++)
            t.Data[i] = (float)(rng.NextDouble() * 2 * limit - limit);
        return t;
    }

    /// <summary>Kaiming / He uniform init (for ReLU networks, conv layers).</summary>
    public static Tensor KaimingUniform(Random rng, int fanIn, params int[] shape)
    {
        float limit = MathF.Sqrt(6f / fanIn);
        var t = new Tensor(shape);
        for (int i = 0; i < t.Length; i++)
            t.Data[i] = (float)(rng.NextDouble() * 2 * limit - limit);
        return t;
    }

    // ───── basic ops (static, produce new tensor) ─────

    /// <summary>Matrix multiply: (M,K) x (K,N) -> (M,N).</summary>
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("MatMul requires 2D tensors");
        int m = a.Shape[0], k = a.Shape[1], n = b.Shape[1];
        if (k != b.Shape[0])
            throw new ArgumentException($"Shape mismatch: ({m},{k}) x ({b.Shape[0]},{n})");

        var c = new Tensor([m, n]);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float sum = 0f;
                for (int p = 0; p < k; p++)
                    sum += a.Get2D(i, p) * b.Get2D(p, j);
                c.Set2D(i, j, sum);
            }
        return c;
    }

    /// <summary>Element-wise add. Supports broadcasting bias (1,N) to (M,N).</summary>
    public static Tensor Add(Tensor a, Tensor b)
    {
        // Simple broadcast: b has fewer elements and its shape matches trailing dims of a
        if (a.Length == b.Length)
        {
            var c = new Tensor((int[])a.Shape.Clone());
            for (int i = 0; i < a.Length; i++)
                c.Data[i] = a.Data[i] + b.Data[i];
            return c;
        }

        // Bias broadcast for 2D: a=(M,N), b=(1,N) or b=(N,)
        if (a.Rank == 2 && (b.Rank == 1 && b.Shape[0] == a.Shape[1]
                         || b.Rank == 2 && b.Shape[0] == 1 && b.Shape[1] == a.Shape[1]))
        {
            int m = a.Shape[0], n = a.Shape[1];
            var c = new Tensor([m, n]);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    c.Set2D(i, j, a.Get2D(i, j) + b.Data[j]);
            return c;
        }

        // 3D broadcast: a=(B,S,D), b=(D,) -> broadcast over B,S
        if (a.Rank == 3 && b.Rank == 1 && b.Shape[0] == a.Shape[2])
        {
            int B = a.Shape[0], S = a.Shape[1], D = a.Shape[2];
            var c = new Tensor([B, S, D]);
            for (int bi = 0; bi < B; bi++)
                for (int si = 0; si < S; si++)
                    for (int di = 0; di < D; di++)
                        c.Set3D(bi, si, di, a.Get3D(bi, si, di) + b.Data[di]);
            return c;
        }

        // 3D broadcast: a=(B,S,D), b=(1,1,D) -> broadcast over B,S
        if (a.Rank == 3 && b.Rank == 3 && b.Shape[0] == 1 && b.Shape[1] == 1 && b.Shape[2] == a.Shape[2])
        {
            int B = a.Shape[0], S = a.Shape[1], D = a.Shape[2];
            var c = new Tensor([B, S, D]);
            for (int bi = 0; bi < B; bi++)
                for (int si = 0; si < S; si++)
                    for (int di = 0; di < D; di++)
                        c.Set3D(bi, si, di, a.Get3D(bi, si, di) + b.Data[di]);
            return c;
        }

        throw new ArgumentException($"Cannot broadcast shapes [{string.Join(',', a.Shape)}] and [{string.Join(',', b.Shape)}]");
    }

    /// <summary>Element-wise ReLU.</summary>
    public static Tensor ReLU(Tensor a)
    {
        var c = new Tensor((int[])a.Shape.Clone());
        for (int i = 0; i < a.Length; i++)
            c.Data[i] = MathF.Max(0, a.Data[i]);
        return c;
    }

    /// <summary>Element-wise Sigmoid.</summary>
    public static Tensor Sigmoid(Tensor a)
    {
        var c = new Tensor((int[])a.Shape.Clone());
        for (int i = 0; i < a.Length; i++)
            c.Data[i] = 1f / (1f + MathF.Exp(-a.Data[i]));
        return c;
    }

    /// <summary>Element-wise Tanh.</summary>
    public static Tensor Tanh(Tensor a)
    {
        var c = new Tensor((int[])a.Shape.Clone());
        for (int i = 0; i < a.Length; i++)
            c.Data[i] = MathF.Tanh(a.Data[i]);
        return c;
    }

    /// <summary>Softmax along last axis. Works for 1D and 2D.</summary>
    public static Tensor Softmax(Tensor a)
    {
        var c = new Tensor((int[])a.Shape.Clone());
        if (a.Rank == 1)
        {
            SoftmaxSpan(a.Data.AsSpan(), c.Data.AsSpan());
        }
        else if (a.Rank == 2)
        {
            int rows = a.Shape[0], cols = a.Shape[1];
            for (int r = 0; r < rows; r++)
                SoftmaxSpan(a.Data.AsSpan(r * cols, cols), c.Data.AsSpan(r * cols, cols));
        }
        else if (a.Rank == 3)
        {
            int B = a.Shape[0], S = a.Shape[1], D = a.Shape[2];
            for (int b = 0; b < B; b++)
                for (int s = 0; s < S; s++)
                {
                    int offset = (b * S + s) * D;
                    SoftmaxSpan(a.Data.AsSpan(offset, D), c.Data.AsSpan(offset, D));
                }
        }
        else
        {
            throw new NotSupportedException("Softmax supports 1D/2D/3D tensors");
        }
        return c;
    }

    // ───── CNN ops ─────

    /// <summary>2D convolution. input=(N,Cin,H,W), weight=(Cout,Cin,kH,kW), bias=(Cout,) or null.</summary>
    public static Tensor Conv2D(Tensor input, Tensor weight, Tensor? bias, int stride = 1, int padding = 0)
    {
        if (input.Rank != 4) throw new ArgumentException("Conv2D input must be 4D (N,C,H,W)");
        if (weight.Rank != 4) throw new ArgumentException("Conv2D weight must be 4D (Cout,Cin,kH,kW)");
        int N = input.Shape[0], Cin = input.Shape[1], Hin = input.Shape[2], Win = input.Shape[3];
        int Cout = weight.Shape[0], kH = weight.Shape[2], kW = weight.Shape[3];
        if (weight.Shape[1] != Cin)
            throw new ArgumentException($"Channel mismatch: input {Cin}, weight {weight.Shape[1]}");

        int Hout = (Hin + 2 * padding - kH) / stride + 1;
        int Wout = (Win + 2 * padding - kW) / stride + 1;
        var output = new Tensor([N, Cout, Hout, Wout]);

        for (int n = 0; n < N; n++)
            for (int co = 0; co < Cout; co++)
            {
                float b = bias != null ? bias.Data[co] : 0f;
                for (int oh = 0; oh < Hout; oh++)
                    for (int ow = 0; ow < Wout; ow++)
                    {
                        float sum = b;
                        for (int ci = 0; ci < Cin; ci++)
                            for (int kh = 0; kh < kH; kh++)
                                for (int kw = 0; kw < kW; kw++)
                                {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win)
                                        sum += input.Get4D(n, ci, ih, iw) * weight.Get4D(co, ci, kh, kw);
                                }
                        output.Set4D(n, co, oh, ow, sum);
                    }
            }
        return output;
    }

    /// <summary>Max pooling 2D. input=(N,C,H,W).</summary>
    public static Tensor MaxPool2D(Tensor input, int kernelSize, int stride = -1)
    {
        if (input.Rank != 4) throw new ArgumentException("MaxPool2D input must be 4D");
        if (stride < 0) stride = kernelSize;
        int N = input.Shape[0], C = input.Shape[1], Hin = input.Shape[2], Win = input.Shape[3];
        int Hout = (Hin - kernelSize) / stride + 1;
        int Wout = (Win - kernelSize) / stride + 1;
        var output = new Tensor([N, C, Hout, Wout]);

        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int oh = 0; oh < Hout; oh++)
                    for (int ow = 0; ow < Wout; ow++)
                    {
                        float max = float.MinValue;
                        for (int kh = 0; kh < kernelSize; kh++)
                            for (int kw = 0; kw < kernelSize; kw++)
                                max = MathF.Max(max, input.Get4D(n, c, oh * stride + kh, ow * stride + kw));
                        output.Set4D(n, c, oh, ow, max);
                    }
        return output;
    }

    /// <summary>Global average pooling: (N,C,H,W) → (N,C).</summary>
    public static Tensor GlobalAvgPool2D(Tensor input)
    {
        if (input.Rank != 4) throw new ArgumentException("GlobalAvgPool2D input must be 4D");
        int N = input.Shape[0], C = input.Shape[1], H = input.Shape[2], W = input.Shape[3];
        var output = new Tensor([N, C]);
        float area = H * W;
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
            {
                float sum = 0;
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        sum += input.Get4D(n, c, h, w);
                output.Set2D(n, c, sum / area);
            }
        return output;
    }

    /// <summary>Batch normalization 2D (inference mode). input=(N,C,H,W).</summary>
    public static Tensor BatchNorm2D(Tensor input, Tensor gamma, Tensor beta,
        Tensor runningMean, Tensor runningVar, float eps = 1e-5f)
    {
        if (input.Rank != 4) throw new ArgumentException("BatchNorm2D input must be 4D");
        int N = input.Shape[0], C = input.Shape[1], H = input.Shape[2], W = input.Shape[3];
        var output = new Tensor([N, C, H, W]);
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
            {
                float mean = runningMean.Data[c];
                float invStd = 1f / MathF.Sqrt(runningVar.Data[c] + eps);
                float g = gamma.Data[c], b = beta.Data[c];
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                    {
                        float val = (input.Get4D(n, c, h, w) - mean) * invStd;
                        output.Set4D(n, c, h, w, g * val + b);
                    }
            }
        return output;
    }

    // ───── Transformer ops ─────

    /// <summary>Batched matrix multiply: (B,M,K) × (B,K,N) → (B,M,N).</summary>
    public static Tensor BatchedMatMul(Tensor a, Tensor b)
    {
        if (a.Rank != 3 || b.Rank != 3)
            throw new ArgumentException("BatchedMatMul requires 3D tensors");
        int B = a.Shape[0], M = a.Shape[1], K = a.Shape[2], N = b.Shape[2];
        if (B != b.Shape[0] || K != b.Shape[1])
            throw new ArgumentException("Shape mismatch for batched matmul");
        var c = new Tensor([B, M, N]);
        for (int bi = 0; bi < B; bi++)
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < K; p++)
                        sum += a.Get3D(bi, i, p) * b.Get3D(bi, p, j);
                    c.Set3D(bi, i, j, sum);
                }
        return c;
    }

    /// <summary>Layer normalization along last axis. gamma/beta shape = (lastDim,).</summary>
    public static Tensor LayerNorm(Tensor input, Tensor gamma, Tensor beta, float eps = 1e-5f)
    {
        int lastDim = input.Shape[^1];
        int outerSize = input.Length / lastDim;
        var output = new Tensor((int[])input.Shape.Clone());
        for (int i = 0; i < outerSize; i++)
        {
            int offset = i * lastDim;
            float mean = 0;
            for (int j = 0; j < lastDim; j++)
                mean += input.Data[offset + j];
            mean /= lastDim;
            float var_ = 0;
            for (int j = 0; j < lastDim; j++)
            {
                float diff = input.Data[offset + j] - mean;
                var_ += diff * diff;
            }
            var_ /= lastDim;
            float invStd = 1f / MathF.Sqrt(var_ + eps);
            for (int j = 0; j < lastDim; j++)
                output.Data[offset + j] = gamma.Data[j] * (input.Data[offset + j] - mean) * invStd + beta.Data[j];
        }
        return output;
    }

    // ───── shape ops ─────

    /// <summary>Transpose 2D tensor.</summary>
    public static Tensor Transpose2D(Tensor a)
    {
        if (a.Rank != 2) throw new ArgumentException("Transpose2D requires 2D tensor");
        int m = a.Shape[0], n = a.Shape[1];
        var c = new Tensor([n, m]);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                c.Set2D(j, i, a.Get2D(i, j));
        return c;
    }

    /// <summary>Reshape (must keep same total length).</summary>
    public Tensor Reshape(params int[] newShape)
    {
        if (ShapeToLength(newShape) != Length)
            throw new ArgumentException("Reshape length mismatch");
        return new Tensor(newShape, Data); // shares data intentionally for read-only trace
    }

    // ───── display ─────

    public override string ToString()
    {
        string shapeStr = $"[{string.Join(',', Shape)}]";
        if (Length <= 20)
            return $"Tensor{shapeStr} [{string.Join(", ", Data.Select(v => v.ToString("F4")))}]";
        return $"Tensor{shapeStr} [{string.Join(", ", Data.Take(5).Select(v => v.ToString("F4")))} ... {string.Join(", ", Data.Skip(Length - 3).Select(v => v.ToString("F4")))}]";
    }

    /// <summary>Deep copy.</summary>
    public Tensor Clone() => new((int[])Shape.Clone(), (float[])Data.Clone());

    // ───── internal ─────

    private static int ShapeToLength(int[] shape)
    {
        int len = 1;
        foreach (int s in shape) len *= s;
        return len;
    }

    private static void SoftmaxSpan(ReadOnlySpan<float> src, Span<float> dst)
    {
        float max = float.MinValue;
        for (int i = 0; i < src.Length; i++)
            if (src[i] > max) max = src[i];

        float sum = 0;
        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = MathF.Exp(src[i] - max);
            sum += dst[i];
        }
        for (int i = 0; i < dst.Length; i++)
            dst[i] /= sum;
    }
}
