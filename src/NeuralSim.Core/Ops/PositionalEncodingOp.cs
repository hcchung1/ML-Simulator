namespace NeuralSim.Core.Ops;

/// <summary>
/// Sinusoidal positional encoding: output = input + PE.
/// Input port:  "input"  — shape (batch, seq, d_model)
/// Output port: "output" — shape (batch, seq, d_model)
/// Parameters:  "pe" (maxSeqLen, d_model) — precomputed sinusoidal table
/// </summary>
public sealed class PositionalEncodingOp : Op
{
    public override string OpType => "PositionalEncoding";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public int DModel { get; }
    public int MaxSeqLen { get; }

    public PositionalEncodingOp(string id, int dModel, int maxSeqLen = 512, string? name = null)
        : base(id, name)
    {
        DModel = dModel;
        MaxSeqLen = maxSeqLen;

        // Precompute sinusoidal positional encoding table
        var pe = new Tensor([maxSeqLen, dModel]);
        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            for (int i = 0; i < dModel; i++)
            {
                double angle = pos / Math.Pow(10000, (2.0 * (i / 2)) / dModel);
                pe.Set2D(pos, i, (float)(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle)));
            }
        }
        Parameters["pe"] = pe;
    }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"]; // (B, S, D)
        int B = x.Shape[0], S = x.Shape[1], D = x.Shape[2];
        var pe = Parameters["pe"]; // (maxSeqLen, D)
        var output = new Tensor([B, S, D]);

        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
                for (int d = 0; d < D; d++)
                    output.Set3D(b, s, d, x.Get3D(b, s, d) + pe.Get2D(s, d));

        return new() { ["output"] = output };
    }

    public override string Describe() => $"PositionalEncoding(d={DModel}, max={MaxSeqLen})";
}
