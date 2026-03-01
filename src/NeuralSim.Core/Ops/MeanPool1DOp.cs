namespace NeuralSim.Core.Ops;

/// <summary>
/// Mean pooling over sequence dimension: (B, S, D) → (B, D).
/// Used to aggregate transformer output for classification.
/// </summary>
public sealed class MeanPool1DOp : Op
{
    public override string OpType => "MeanPool1D";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public MeanPool1DOp(string id, string? name = null) : base(id, name) { }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"]; // (B, S, D)
        if (x.Rank != 3)
            throw new ArgumentException("MeanPool1D expects 3D input (B, S, D)");

        int B = x.Shape[0], S = x.Shape[1], D = x.Shape[2];
        var output = new Tensor([B, D]);

        for (int b = 0; b < B; b++)
            for (int d = 0; d < D; d++)
            {
                float sum = 0;
                for (int s = 0; s < S; s++)
                    sum += x.Get3D(b, s, d);
                output.Set2D(b, d, sum / S);
            }

        return new() { ["output"] = output };
    }

    public override string Describe() => "MeanPool1D (seq→avg)";
}
