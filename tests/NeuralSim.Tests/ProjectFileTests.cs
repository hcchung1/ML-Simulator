using NeuralSim.Core;
using NeuralSim.Core.Builders;
using Xunit;

namespace NeuralSim.Tests;

public class ProjectFileTests
{
    [Fact]
    public void ProjectFile_RoundTrip_SerializeDeserialize()
    {
        var project = new ProjectFile
        {
            Name = "Test Model",
            InputText = "1.0, 2.0, 3.0",
            Seed = 42,
            Nodes =
            [
                new ProjectNode { Id = "input", OpType = "GraphInput", X = 100, Y = 20, IsFixed = true },
                new ProjectNode { Id = "conv1", OpType = "Conv2D", X = 100, Y = 100, InChannels = 1, OutChannels = 16, KernelSize = 3, Stride = 1, Padding = 1 },
                new ProjectNode { Id = "output", OpType = "GraphOutput", X = 100, Y = 400, IsFixed = true },
            ],
            Connections =
            [
                new ProjectConnection { FromId = "input", FromPort = "output", ToId = "conv1", ToPort = "input" },
                new ProjectConnection { FromId = "conv1", FromPort = "output", ToId = "output", ToPort = "input" },
            ],
        };

        var json = project.ToJson();
        Assert.Contains("Test Model", json);

        var loaded = ProjectFile.FromJson(json);
        Assert.Equal("Test Model", loaded.Name);
        Assert.Equal("1.0, 2.0, 3.0", loaded.InputText);
        Assert.Equal(42, loaded.Seed);
        Assert.Equal(3, loaded.Nodes.Count);
        Assert.Equal(2, loaded.Connections.Count);
        Assert.Equal("Conv2D", loaded.Nodes[1].OpType);
        Assert.Equal(16, loaded.Nodes[1].OutChannels);
    }

    [Fact]
    public void ProjectFile_TransformerConfig_Preserved()
    {
        var project = new ProjectFile
        {
            Nodes =
            [
                new ProjectNode { Id = "mha", OpType = "MultiHeadAttention", DModel = 64, NumHeads = 4 },
            ]
        };

        var json = project.ToJson();
        var loaded = ProjectFile.FromJson(json);

        Assert.Equal(64, loaded.Nodes[0].DModel);
        Assert.Equal(4, loaded.Nodes[0].NumHeads);
    }
}
