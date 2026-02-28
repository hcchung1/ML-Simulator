namespace NeuralSim.Core.Ops;

/// <summary>
/// Element-wise Sigmoid activation: 1/(1+exp(-x)).
/// </summary>
public sealed class SigmoidOp : Op
{
    public override string OpType => "Sigmoid";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public SigmoidOp(string id, string? name = null) : base(id, name) { }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"];
        return new() { ["output"] = Tensor.Sigmoid(x) };
    }
}
