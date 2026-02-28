using NeuralSim.Core.Ops;

namespace NeuralSim.Core;

/// <summary>
/// Runs a <see cref="Graph"/> on an input tensor and produces a <see cref="Trace"/>
/// recording every intermediate result for step-by-step replay.
/// </summary>
public static class Executor
{
    /// <summary>
    /// Execute the graph forward pass, recording a full trace.
    /// </summary>
    /// <param name="graph">The compute graph to run.</param>
    /// <param name="input">Input tensor (e.g. a single sample or a batch).</param>
    /// <returns>A trace containing every step's inputs, outputs, and parameters.</returns>
    public static Trace Run(Graph graph, Tensor input)
    {
        var trace = new Trace { Input = input.Clone() };

        // Tensor store: opId:portName → tensor
        var store = new Dictionary<string, Tensor>
        {
            ["graph_input:output"] = input
        };

        var order = graph.TopologicalOrder();

        for (int i = 0; i < order.Count; i++)
        {
            var op = order[i];

            // Resolve input tensors for this op
            var opInputs = new Dictionary<string, Tensor>();
            foreach (var (portName, source) in op.InputPorts)
            {
                string key = $"{source.OpId}:{source.PortName}";
                if (!store.TryGetValue(key, out var tensor))
                    throw new InvalidOperationException(
                        $"Op '{op.Id}' input port '{portName}' references '{key}' which is not yet computed.");
                opInputs[portName] = tensor;
            }

            // Execute
            var opOutputs = op.Compute(opInputs);

            // Store outputs
            foreach (var (portName, tensor) in opOutputs)
                store[$"{op.Id}:{portName}"] = tensor;

            // Record trace step (clone tensors so trace is immutable)
            var step = new TraceStep
            {
                Index = i,
                OpId = op.Id,
                OpName = op.Name,
                OpType = op.OpType,
                Description = op.Describe(),
                Inputs = opInputs.ToDictionary(kv => kv.Key, kv => kv.Value.Clone()),
                Outputs = opOutputs.ToDictionary(kv => kv.Key, kv => kv.Value.Clone()),
                Parameters = op.Parameters.ToDictionary(kv => kv.Key, kv => kv.Value.Clone())
            };
            trace.Steps.Add(step);
        }

        // Final output = last op's first output
        if (order.Count > 0)
        {
            var lastOp = order[^1];
            string lastKey = $"{lastOp.Id}:{lastOp.OutputNames[0]}";
            trace.Output = store[lastKey].Clone();
        }

        return trace;
    }
}
