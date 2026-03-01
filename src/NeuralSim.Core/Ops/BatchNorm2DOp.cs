namespace NeuralSim.Core.Ops;

/// <summary>
/// Batch normalization 2D (inference mode).
/// Input port:  "input"  — shape (N, C, H, W)
/// Output port: "output" — shape (N, C, H, W)
/// Parameters:  "gamma" (C,), "beta" (C,), "running_mean" (C,), "running_var" (C,)
/// </summary>
public sealed class BatchNorm2DOp : Op
{
    public override string OpType => "BatchNorm2D";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public int NumFeatures { get; }

    public BatchNorm2DOp(string id, int numFeatures, Random? rng = null, string? name = null)
        : base(id, name)
    {
        NumFeatures = numFeatures;

        // gamma=1, beta=0, running stats = standard normal
        var gamma = new Tensor([numFeatures]);
        Array.Fill(gamma.Data, 1f);
        Parameters["gamma"] = gamma;
        Parameters["beta"] = Tensor.Zeros(numFeatures);
        Parameters["running_mean"] = Tensor.Zeros(numFeatures);
        var runVar = new Tensor([numFeatures]);
        Array.Fill(runVar.Data, 1f);
        Parameters["running_var"] = runVar;
    }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"];
        return new()
        {
            ["output"] = Tensor.BatchNorm2D(x,
                Parameters["gamma"], Parameters["beta"],
                Parameters["running_mean"], Parameters["running_var"])
        };
    }

    public override string Describe() => $"BatchNorm2D({NumFeatures})";
}
