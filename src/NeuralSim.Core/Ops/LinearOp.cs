namespace NeuralSim.Core.Ops;

/// <summary>
/// Fully-connected linear layer: output = input @ weight + bias.
/// Input port:  "input"  — shape (batch, inFeatures)
/// Output port: "output" — shape (batch, outFeatures)
/// Parameters:  "weight" (inFeatures, outFeatures), "bias" (outFeatures,)
/// </summary>
public sealed class LinearOp : Op
{
    public override string OpType => "Linear";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public int InFeatures { get; }
    public int OutFeatures { get; }

    public LinearOp(string id, int inFeatures, int outFeatures, Random? rng = null, string? name = null)
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
        var x = inputs["input"]; // (batch, inFeatures)
        var w = Parameters["weight"]; // (inFeatures, outFeatures)
        var b = Parameters["bias"]; // (outFeatures,)

        var z = Tensor.MatMul(x, w);
        z = Tensor.Add(z, b);

        return new() { ["output"] = z };
    }

    public override string Describe() =>
        $"Linear({InFeatures} → {OutFeatures})";
}
