namespace NeuralSim.Core.Ops;

/// <summary>
/// Layer normalization along last axis.
/// Input port:  "input"
/// Output port: "output"
/// Parameters:  "gamma" (D,), "beta" (D,)
/// </summary>
public sealed class LayerNormOp : Op
{
    public override string OpType => "LayerNorm";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public int NormSize { get; }

    public LayerNormOp(string id, int normSize, string? name = null) : base(id, name)
    {
        NormSize = normSize;
        var gamma = new Tensor([normSize]);
        Array.Fill(gamma.Data, 1f);
        Parameters["gamma"] = gamma;
        Parameters["beta"] = Tensor.Zeros(normSize);
    }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"];
        return new() { ["output"] = Tensor.LayerNorm(x, Parameters["gamma"], Parameters["beta"]) };
    }

    public override string Describe() => $"LayerNorm({NormSize})";
}
