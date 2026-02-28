using NeuralSim.Core;
using NeuralSim.Core.Ops;
using NeuralSim.Core.Builders;
using Xunit;

namespace NeuralSim.Tests;

public class MlpTests
{
    private const float Eps = 1e-4f;

    [Fact]
    public void LinearOp_ManualWeights_CorrectOutput()
    {
        // input=[1,2], W=[[0.1,0.3],[0.2,0.4]], b=[0.5,0.6]
        // z = [1,2] @ W + b = [0.1+0.4+0.5, 0.3+0.8+0.6] = [1.0, 1.7]
        var linear = new LinearOp("l0", 2, 2);
        linear.Parameters["weight"] = new Tensor([2, 2], [0.1f, 0.3f, 0.2f, 0.4f]);
        linear.Parameters["bias"] = new Tensor([2], [0.5f, 0.6f]);

        var input = new Tensor([1, 2], [1f, 2f]);
        var result = linear.Compute(new() { ["input"] = input });
        var output = result["output"];

        Assert.Equal([1, 2], output.Shape);
        Assert.Equal(1.0f, output.Get2D(0, 0), Eps);
        Assert.Equal(1.7f, output.Get2D(0, 1), Eps);
    }

    [Fact]
    public void MlpBuilder_Creates_CorrectNumberOfOps()
    {
        // 3 -> 4 -> 2: should be Linear+ReLU, Linear+Softmax = 4 ops
        var graph = MlpBuilder.Build([3, 4, 2]);
        Assert.Equal(4, graph.Nodes.Count);
        Assert.Equal("Linear", graph.Nodes[0].OpType);
        Assert.Equal("ReLU", graph.Nodes[1].OpType);
        Assert.Equal("Linear", graph.Nodes[2].OpType);
        Assert.Equal("Softmax", graph.Nodes[3].OpType);
    }

    [Fact]
    public void Executor_MLP_ProducesTrace_WithCorrectSteps()
    {
        var graph = MlpBuilder.Build([3, 4, 2], seed: 123);
        var input = new Tensor([1, 3], [0.5f, -0.3f, 0.8f]);

        var trace = Executor.Run(graph, input);

        Assert.Equal(4, trace.StepCount);
        Assert.NotNull(trace.Output);
        Assert.Equal([1, 2], trace.Output!.Shape);

        // Softmax output should sum to ~1
        float sum = trace.Output.Data.Sum();
        Assert.Equal(1f, sum, Eps);

        // Each step should have inputs and outputs recorded
        foreach (var step in trace.Steps)
        {
            Assert.NotEmpty(step.Inputs);
            Assert.NotEmpty(step.Outputs);
        }
    }

    [Fact]
    public void Executor_MLP_HandComputed_2x2()
    {
        // Carefully hand-computed MLP: 2 -> 2 -> 1
        // Linear0: W=[[0.1, 0.3],[0.2, 0.4]], b=[0, 0]
        // ReLU
        // Linear1: W=[[0.5],[0.6]], b=[0.1]

        var l0 = new LinearOp("l0", 2, 2);
        l0.Parameters["weight"] = new Tensor([2, 2], [0.1f, 0.3f, 0.2f, 0.4f]);
        l0.Parameters["bias"] = new Tensor([2], [0f, 0f]);

        var relu = new ReLUOp("relu0");

        var l1 = new LinearOp("l1", 2, 1);
        l1.Parameters["weight"] = new Tensor([2, 1], [0.5f, 0.6f]);
        l1.Parameters["bias"] = new Tensor([1], [0.1f]);

        var graph = Graph.Sequential(l0, relu, l1);
        var input = new Tensor([1, 2], [1f, 2f]);
        var trace = Executor.Run(graph, input);

        // Step 0: Linear0
        // z0 = [1,2] @ [[0.1,0.3],[0.2,0.4]] + [0,0] = [0.5, 1.1]
        var step0out = trace.Steps[0].Outputs["output"];
        Assert.Equal(0.5f, step0out.Get2D(0, 0), Eps);
        Assert.Equal(1.1f, step0out.Get2D(0, 1), Eps);

        // Step 1: ReLU
        // relu([0.5, 1.1]) = [0.5, 1.1] (both positive)
        var step1out = trace.Steps[1].Outputs["output"];
        Assert.Equal(0.5f, step1out.Get2D(0, 0), Eps);
        Assert.Equal(1.1f, step1out.Get2D(0, 1), Eps);

        // Step 2: Linear1
        // z1 = [0.5, 1.1] @ [[0.5],[0.6]] + [0.1] = [0.5*0.5 + 1.1*0.6 + 0.1] = [0.25+0.66+0.1] = [1.01]
        var step2out = trace.Steps[2].Outputs["output"];
        Assert.Equal(1.01f, step2out.Get2D(0, 0), Eps);
    }

    [Fact]
    public void Graph_TopologicalOrder_SequentialIsCorrect()
    {
        var l0 = new LinearOp("a", 2, 3);
        var relu = new ReLUOp("b");
        var l1 = new LinearOp("c", 3, 1);
        var graph = Graph.Sequential(l0, relu, l1);

        var order = graph.TopologicalOrder();
        Assert.Equal("a", order[0].Id);
        Assert.Equal("b", order[1].Id);
        Assert.Equal("c", order[2].Id);
    }
}
