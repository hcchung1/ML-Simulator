using NeuralSim.Core;
using NeuralSim.Core.Ops;
using NeuralSim.Core.Builders;
using Xunit;

namespace NeuralSim.Tests;

public class ResNetTests
{
    private const float Eps = 1e-4f;

    [Fact]
    public void ResNetBuilder_Build_ProducesValidGraph()
    {
        var graph = ResNetBuilder.Build(
            inputChannels: 1, imageSize: 8,
            channels: [4, 8], blocksPerStage: 1,
            numClasses: 3, seed: 42);

        Assert.True(graph.Nodes.Count > 0);
        // Should be able to compute topological order without cycles
        var order = graph.TopologicalOrder();
        Assert.True(order.Count > 0);
    }

    [Fact]
    public void ResNetBuilder_ExecutesForwardPass()
    {
        var graph = ResNetBuilder.Build(
            inputChannels: 1, imageSize: 8,
            channels: [4, 8], blocksPerStage: 1,
            numClasses: 3, seed: 42);

        var input = Tensor.Random(new Random(0), 1, 1, 8, 8);
        var trace = Executor.Run(graph, input);

        Assert.NotNull(trace.Output);
        Assert.Equal([1, 3], trace.Output!.Shape);

        // Softmax output sums to ~1
        float sum = trace.Output.Data.Sum();
        Assert.Equal(1f, sum, Eps);
    }

    [Fact]
    public void ResNetBuilder_HasResidualConnections()
    {
        var graph = ResNetBuilder.Build(
            inputChannels: 1, imageSize: 8,
            channels: [4], blocksPerStage: 1,
            numClasses: 2, seed: 42);

        // Should contain at least one Add op (for residual connection)
        bool hasAdd = graph.Nodes.Any(n => n.OpType == "Add");
        Assert.True(hasAdd, "ResNet should have Add ops for skip connections");
    }

    [Fact]
    public void ResNetBuilder_AllStepsProduceOutputs()
    {
        var graph = ResNetBuilder.Build(
            inputChannels: 1, imageSize: 8,
            channels: [4], blocksPerStage: 1,
            numClasses: 2, seed: 42);

        var input = Tensor.Random(new Random(0), 1, 1, 8, 8);
        var trace = Executor.Run(graph, input);

        foreach (var step in trace.Steps)
        {
            Assert.NotEmpty(step.Outputs);
            foreach (var (_, tensor) in step.Outputs)
                Assert.True(tensor.Length > 0);
        }
    }
}
