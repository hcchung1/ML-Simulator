namespace NeuralSim.Core.Ops;

/// <summary>
/// Softmax along last axis.
/// </summary>
public sealed class SoftmaxOp : Op
{
    public override string OpType => "Softmax";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public SoftmaxOp(string id, string? name = null) : base(id, name) { }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"];
        return new() { ["output"] = Tensor.Softmax(x) };
    }
}
