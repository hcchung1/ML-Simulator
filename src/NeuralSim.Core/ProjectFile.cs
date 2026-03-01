using System.Text.Json;
using System.Text.Json.Serialization;

namespace NeuralSim.Core;

/// <summary>
/// Serializable project file for saving/loading NeuralSim models.
/// File extension: .nsim
/// </summary>
public sealed class ProjectFile
{
    public string Version { get; set; } = "1.1";
    public string Name { get; set; } = "Untitled";
    public string Description { get; set; } = "";
    public string InputText { get; set; } = "";
    public int Seed { get; set; } = 42;
    public List<ProjectNode> Nodes { get; set; } = [];
    public List<ProjectConnection> Connections { get; set; } = [];

    private static readonly JsonSerializerOptions _options = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingDefault,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    public string ToJson() => JsonSerializer.Serialize(this, _options);

    public static ProjectFile FromJson(string json) =>
        JsonSerializer.Deserialize<ProjectFile>(json, _options)
        ?? throw new InvalidOperationException("Invalid project file");
}

/// <summary>A single node in the saved project.</summary>
public sealed class ProjectNode
{
    public string Id { get; set; } = "";
    public string OpType { get; set; } = "";
    public string Label { get; set; } = "";
    public double X { get; set; }
    public double Y { get; set; }
    public bool IsFixed { get; set; }

    // Linear
    public int InFeatures { get; set; }
    public int OutFeatures { get; set; }

    // Conv2D / MaxPool2D / BatchNorm2D
    public int InChannels { get; set; }
    public int OutChannels { get; set; }
    public int KernelSize { get; set; }
    public int Stride { get; set; }
    public int Padding { get; set; }

    // Transformer
    public int NumHeads { get; set; }
    public int DModel { get; set; }
    public int MaxSeqLen { get; set; }
}

/// <summary>A single connection in the saved project.</summary>
public sealed class ProjectConnection
{
    public string FromId { get; set; } = "";
    public string FromPort { get; set; } = "output";
    public string ToId { get; set; } = "";
    public string ToPort { get; set; } = "input";
}
