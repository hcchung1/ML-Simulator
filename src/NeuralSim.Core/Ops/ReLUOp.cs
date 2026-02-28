namespace NeuralSim.Core.Ops;

/// <summary>
/// Element-wise ReLU activation: max(0, x).
/// </summary>
public sealed class ReLUOp : Op
{
    public override string OpType => "ReLU";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public ReLUOp(string id, string? name = null) : base(id, name) { }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"];
        return new() { ["output"] = Tensor.ReLU(x) };
    }
}
