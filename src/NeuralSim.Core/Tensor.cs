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
        else
        {
            throw new NotSupportedException("Softmax currently supports 1D/2D tensors");
        }
        return c;
    }

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
