namespace NeuralSim.Core.Ops;

/// <summary>
/// Multi-head self-attention.
/// Input port:  "input"  — shape (batch, seq, d_model)
/// Output port: "output" — shape (batch, seq, d_model)
/// Parameters:  Wq, Wk, Wv, Wo (d_model, d_model), bq, bk, bv, bo (d_model,)
/// </summary>
public sealed class MultiHeadAttentionOp : Op
{
    public override string OpType => "MultiHeadAttention";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public int DModel { get; }
    public int NumHeads { get; }
    public int Dk => DModel / NumHeads;

    public MultiHeadAttentionOp(string id, int dModel, int numHeads, Random? rng = null, string? name = null)
        : base(id, name)
    {
        if (dModel % numHeads != 0)
            throw new ArgumentException($"d_model ({dModel}) must be divisible by num_heads ({numHeads})");

        DModel = dModel;
        NumHeads = numHeads;
        rng ??= Random.Shared;

        Parameters["Wq"] = Tensor.XavierUniform(rng, dModel, dModel);
        Parameters["Wk"] = Tensor.XavierUniform(rng, dModel, dModel);
        Parameters["Wv"] = Tensor.XavierUniform(rng, dModel, dModel);
        Parameters["Wo"] = Tensor.XavierUniform(rng, dModel, dModel);
        Parameters["bq"] = Tensor.Zeros(dModel);
        Parameters["bk"] = Tensor.Zeros(dModel);
        Parameters["bv"] = Tensor.Zeros(dModel);
        Parameters["bo"] = Tensor.Zeros(dModel);
    }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"]; // (B, S, D)
        int B = x.Shape[0], S = x.Shape[1], D = x.Shape[2];
        int dk = Dk;

        // Linear projections: reshape (B,S,D) -> (B*S, D), matmul, reshape back
        var xFlat = x.Reshape(B * S, D);
        var Q = Tensor.Add(Tensor.MatMul(xFlat, Parameters["Wq"]), Parameters["bq"]).Reshape(B, S, D);
        var K = Tensor.Add(Tensor.MatMul(xFlat, Parameters["Wk"]), Parameters["bk"]).Reshape(B, S, D);
        var V = Tensor.Add(Tensor.MatMul(xFlat, Parameters["Wv"]), Parameters["bv"]).Reshape(B, S, D);

        // Multi-head attention
        var output = new Tensor([B, S, D]);
        float scale = MathF.Sqrt(dk);

        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < NumHeads; h++)
            {
                int hOff = h * dk;

                // Attention scores: Q_h @ K_h^T / sqrt(dk) → (S, S)
                var scores = new float[S * S];
                for (int i = 0; i < S; i++)
                    for (int j = 0; j < S; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < dk; k++)
                            sum += Q.Get3D(b, i, hOff + k) * K.Get3D(b, j, hOff + k);
                        scores[i * S + j] = sum / scale;
                    }

                // Softmax each row
                for (int i = 0; i < S; i++)
                {
                    float max = float.MinValue;
                    for (int j = 0; j < S; j++)
                        if (scores[i * S + j] > max) max = scores[i * S + j];
                    float expSum = 0;
                    for (int j = 0; j < S; j++)
                    {
                        scores[i * S + j] = MathF.Exp(scores[i * S + j] - max);
                        expSum += scores[i * S + j];
                    }
                    for (int j = 0; j < S; j++)
                        scores[i * S + j] /= expSum;
                }

                // Weighted sum: scores @ V_h → (S, dk)
                for (int i = 0; i < S; i++)
                    for (int k = 0; k < dk; k++)
                    {
                        float sum = 0;
                        for (int j = 0; j < S; j++)
                            sum += scores[i * S + j] * V.Get3D(b, j, hOff + k);
                        output.Set3D(b, i, hOff + k, sum);
                    }
            }
        }

        // Output projection
        var outFlat = output.Reshape(B * S, D);
        var result = Tensor.Add(Tensor.MatMul(outFlat, Parameters["Wo"]), Parameters["bo"]).Reshape(B, S, D);

        return new() { ["output"] = result };
    }

    public override string Describe() => $"MultiHeadAttention(d={DModel}, h={NumHeads})";
}
