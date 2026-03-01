namespace NeuralSim.Core.Ops;

/// <summary>
/// 2D convolution: output = conv2d(input, weight) + bias.
/// Input port:  "input"  — shape (N, Cin, H, W)
/// Output port: "output" — shape (N, Cout, Hout, Wout)
/// Parameters:  "weight" (Cout, Cin, kH, kW), "bias" (Cout,)
/// </summary>
public sealed class Conv2DOp : Op
{
    public override string OpType => "Conv2D";
    public override IReadOnlyList<string> OutputNames { get; } = ["output"];

    public int InChannels { get; }
    public int OutChannels { get; }
    public int KernelSize { get; }
    public int Stride { get; }
    public int Padding { get; }

    public Conv2DOp(string id, int inChannels, int outChannels, int kernelSize,
                    int stride = 1, int padding = 0, Random? rng = null, string? name = null)
        : base(id, name)
    {
        InChannels = inChannels;
        OutChannels = outChannels;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;
        rng ??= Random.Shared;

        int fanIn = inChannels * kernelSize * kernelSize;
        Parameters["weight"] = Tensor.KaimingUniform(rng, fanIn,
            outChannels, inChannels, kernelSize, kernelSize);
        Parameters["bias"] = Tensor.Zeros(outChannels);
    }

    public override Dictionary<string, Tensor> Compute(Dictionary<string, Tensor> inputs)
    {
        var x = inputs["input"];
        var w = Parameters["weight"];
        var b = Parameters["bias"];
        return new() { ["output"] = Tensor.Conv2D(x, w, b, Stride, Padding) };
    }

    public override string Describe() =>
        $"Conv2D({InChannels}→{OutChannels}, k={KernelSize}, s={Stride}, p={Padding})";
}
