using NeuralSim.Core;
using NeuralSim.Core.Ops;
using NeuralSim.Core.Builders;
using Xunit;

namespace NeuralSim.Tests;

public class CnnTests
{
    private const float Eps = 1e-4f;

    // ───── Tensor CNN ops ─────

    [Fact]
    public void Conv2D_1x1x4x4_SingleFilter_NoPadding_CorrectShape()
    {
        // input=(1,1,4,4), weight=(1,1,3,3), stride=1, padding=0 → output=(1,1,2,2)
        var input = Tensor.Random(new Random(0), 1, 1, 4, 4);
        var weight = new Tensor([1, 1, 3, 3]);
        Array.Fill(weight.Data, 1f); // all-ones filter = sum of 3x3 patch
        var bias = Tensor.Zeros(1);

        var output = Tensor.Conv2D(input, weight, bias, stride: 1, padding: 0);
        Assert.Equal([1, 1, 2, 2], output.Shape);
    }

    [Fact]
    public void Conv2D_WithPadding_PreservesSize()
    {
        // input=(1,1,4,4), k=3, s=1, p=1 → output=(1,1,4,4)
        var input = Tensor.Random(new Random(0), 1, 1, 4, 4);
        var weight = new Tensor([2, 1, 3, 3]);
        Array.Fill(weight.Data, 0.1f);
        var bias = Tensor.Zeros(2);

        var output = Tensor.Conv2D(input, weight, bias, stride: 1, padding: 1);
        Assert.Equal([1, 2, 4, 4], output.Shape);
    }

    [Fact]
    public void Conv2D_ManualCompute_CenterPixel()
    {
        // 1x1x3x3 input, 1x1x3x3 all-ones filter, no padding
        // output should be sum of all 9 input values
        var input = new Tensor([1, 1, 3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        var weight = new Tensor([1, 1, 3, 3]);
        Array.Fill(weight.Data, 1f);

        var output = Tensor.Conv2D(input, weight, null, stride: 1, padding: 0);
        Assert.Equal([1, 1, 1, 1], output.Shape);
        Assert.Equal(45f, output.Data[0], Eps); // sum of 1..9
    }

    [Fact]
    public void MaxPool2D_Halves_SpatialSize()
    {
        var input = Tensor.Random(new Random(0), 1, 2, 4, 4);
        var output = Tensor.MaxPool2D(input, kernelSize: 2, stride: 2);
        Assert.Equal([1, 2, 2, 2], output.Shape);
    }

    [Fact]
    public void MaxPool2D_PicksMax()
    {
        // 1x1x2x2 input [1,3,2,4], pool 2x2 → max = 4
        var input = new Tensor([1, 1, 2, 2], [1f, 3f, 2f, 4f]);
        var output = Tensor.MaxPool2D(input, 2);
        Assert.Equal([1, 1, 1, 1], output.Shape);
        Assert.Equal(4f, output.Data[0], Eps);
    }

    [Fact]
    public void GlobalAvgPool2D_CorrectResult()
    {
        // 1x1x2x2 input [1,2,3,4], avg = 2.5
        var input = new Tensor([1, 1, 2, 2], [1f, 2f, 3f, 4f]);
        var output = Tensor.GlobalAvgPool2D(input);
        Assert.Equal([1, 1], output.Shape);
        Assert.Equal(2.5f, output.Data[0], Eps);
    }

    [Fact]
    public void BatchNorm2D_Normalizes()
    {
        // Simple case: mean=0, var=1, gamma=1, beta=0 → output = input
        var input = new Tensor([1, 2, 2, 2]);
        for (int i = 0; i < input.Length; i++) input.Data[i] = i;

        var gamma = new Tensor([2], [1f, 1f]);
        var beta = Tensor.Zeros(2);
        var mean = Tensor.Zeros(2);
        var var_ = new Tensor([2], [1f, 1f]);

        var output = Tensor.BatchNorm2D(input, gamma, beta, mean, var_, eps: 0f);
        Assert.Equal([1, 2, 2, 2], output.Shape);
        // Channel 0: values 0,1,2,3 normalized with mean=0,var=1 → 0,1,2,3
        Assert.Equal(0f, output.Get4D(0, 0, 0, 0), Eps);
        Assert.Equal(1f, output.Get4D(0, 0, 0, 1), Eps);
    }

    // ───── Conv2D Op ─────

    [Fact]
    public void Conv2DOp_Compute_CorrectShape()
    {
        var op = new Conv2DOp("conv0", 1, 8, 3, stride: 1, padding: 1, rng: new Random(42));
        var input = Tensor.Random(new Random(0), 1, 1, 8, 8);
        var result = op.Compute(new() { ["input"] = input });
        Assert.Equal([1, 8, 8, 8], result["output"].Shape);
    }

    [Fact]
    public void FlattenOp_Flattens4D()
    {
        var op = new FlattenOp("flat");
        var input = Tensor.Random(new Random(0), 1, 8, 4, 4);
        var result = op.Compute(new() { ["input"] = input });
        Assert.Equal([1, 128], result["output"].Shape);
    }

    // ───── CNN Builder ─────

    [Fact]
    public void CnnBuilder_Build_ExecutesCorrectly()
    {
        var graph = CnnBuilder.Build(
            inputChannels: 1, imageSize: 8,
            convChannels: [4, 8], numClasses: 3, seed: 42);

        var input = Tensor.Random(new Random(0), 1, 1, 8, 8);
        var trace = Executor.Run(graph, input);

        Assert.NotNull(trace.Output);
        Assert.Equal([1, 3], trace.Output!.Shape);

        // Softmax output sums to ~1
        float sum = trace.Output.Data.Sum();
        Assert.Equal(1f, sum, Eps);
    }

    [Fact]
    public void CnnBuilder_AllStepsHaveOutputs()
    {
        var graph = CnnBuilder.Build(
            inputChannels: 1, imageSize: 8,
            convChannels: [4], numClasses: 2, seed: 42);

        var input = Tensor.Random(new Random(0), 1, 1, 8, 8);
        var trace = Executor.Run(graph, input);

        foreach (var step in trace.Steps)
        {
            Assert.NotEmpty(step.Outputs);
        }
    }
}
