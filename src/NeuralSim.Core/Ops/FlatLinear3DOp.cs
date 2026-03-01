namespace NeuralSim.Core.Ops;

/// <summary>
/// Linear layer that handles 3D input: (B,S,D) → reshape → matmul → reshape back.
/// Used in Transformer feed-forward blocks.
/// </summary>
public sealed class FlatLinear3DOp : Op
{
    public override string OpType => "Linear";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public int InFeatures { get; }
    public int OutFeatures { get; }

    public FlatLinear3DOp(string id, int inFeatures, int outFeatures, Random? rng = null, string? name = null)
        : base(id, name)
    {
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        rng ??= Random.Shared;
        Parameters["weight"] = Tensor.XavierUniform(rng, inFeatures, outFeatures);
        Parameters["bias"] = Tensor.Zeros(outFeatures);
    }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"];
        var w = Parameters["weight"];
        var b = Parameters["bias"];

        if (x.Rank == 3)
        {
            int B = x.Shape[0], S = x.Shape[1], D = x.Shape[2];
            var flat = x.Reshape(B * S, D);
            var z = Tensor.Add(Tensor.MatMul(flat, w), b);
            return new() { ["output"] = z.Reshape(B, S, OutFeatures) };
        }

        var result = Tensor.Add(Tensor.MatMul(x, w), b);
        return new() { ["output"] = result };
    }

    public override string Describe() => $"Linear({InFeatures} → {OutFeatures})";
}
