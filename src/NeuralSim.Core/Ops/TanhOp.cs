namespace NeuralSim.Core.Ops;

/// <summary>
/// Element-wise Tanh activation.
/// </summary>
public sealed class TanhOp : Op
{
    public override string OpType => "Tanh";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public TanhOp(string id, string? name = null) : base(id, name) { }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"];
        return new() { ["output"] = Tensor.Tanh(x) };
    }
}
