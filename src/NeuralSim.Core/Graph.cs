using NeuralSim.Core.Ops;

namespace NeuralSim.Core;

/// <summary>
/// Directed acyclic graph of Ops.
/// Supports arbitrary DAG topology (not limited to sequential chains).
/// MLP can be expressed as a simple chain; ResNet/Transformer use branching + merging.
/// </summary>
public sealed class Graph
{
    private readonly List<Op> _nodes = [];
    private readonly Dictionary<string, Op> _nodeMap = new();
    private List<Op>? _topoOrder;

    /// <summary>All ops in insertion order.</summary>
    public IReadOnlyList<Op> Nodes => _nodes;

    /// <summary>Graph input port name (convention: the first op reads from this virtual source).</summary>
    public string InputPortName { get; set; } = "graph_input";

    /// <summary>
    /// Add an op to the graph. Wiring (InputPorts) should be set on the op before or after adding.
    /// </summary>
    public void AddOp(Op op)
    {
        if (_nodeMap.ContainsKey(op.Id))
            throw new ArgumentException($"Duplicate op id: {op.Id}");
        _nodeMap[op.Id] = op;
        _nodes.Add(op);
        _topoOrder = null; // invalidate cache
    }

    /// <summary>Get op by id.</summary>
    public Op GetOp(string id) => _nodeMap[id];

    /// <summary>
    /// Wire: targetOp.inputPort ← sourceOp.outputPort.
    /// Convenience method so callers don't need to manipulate OpPort directly.
    /// </summary>
    public void Connect(string sourceOpId, string sourcePort, string targetOpId, string targetPort)
    {
        var target = _nodeMap[targetOpId];
        target.InputPorts[targetPort] = new OpPort(sourceOpId, sourcePort);
        _topoOrder = null;
    }

    /// <summary>
    /// Returns ops in topological execution order (Kahn's algorithm).
    /// </summary>
    public IReadOnlyList<Op> TopologicalOrder()
    {
        if (_topoOrder != null) return _topoOrder;

        // Build adjacency: op id -> set of downstream op ids
        var inDegree = new Dictionary<string, int>();
        var downstream = new Dictionary<string, List<string>>();

        foreach (var op in _nodes)
        {
            inDegree.TryAdd(op.Id, 0);
            downstream.TryAdd(op.Id, []);
        }

        foreach (var op in _nodes)
        {
            foreach (var port in op.InputPorts.Values)
            {
                // port.OpId may be "graph_input" (virtual), skip those
                if (!_nodeMap.ContainsKey(port.OpId)) continue;
                downstream[port.OpId].Add(op.Id);
                inDegree[op.Id] = inDegree.GetValueOrDefault(op.Id) + 1;
            }
        }

        var queue = new Queue<string>();
        foreach (var (id, deg) in inDegree)
            if (deg == 0) queue.Enqueue(id);

        _topoOrder = [];
        while (queue.Count > 0)
        {
            var id = queue.Dequeue();
            _topoOrder.Add(_nodeMap[id]);
            foreach (var next in downstream[id])
            {
                inDegree[next]--;
                if (inDegree[next] == 0) queue.Enqueue(next);
            }
        }

        if (_topoOrder.Count != _nodes.Count)
            throw new InvalidOperationException("Graph contains a cycle!");

        return _topoOrder;
    }

    // ───── convenience builder for sequential (chain) graphs ─────

    /// <summary>
    /// Build a simple sequential graph: op[0] → op[1] → ... → op[n-1].
    /// Automatically wires each op's "input" port to the previous op's "output" port.
    /// The first op's "input" comes from "graph_input".
    /// </summary>
    public static Graph Sequential(params Op[] ops)
    {
        var g = new Graph();
        for (int i = 0; i < ops.Length; i++)
        {
            g.AddOp(ops[i]);
            if (i == 0)
            {
                ops[i].InputPorts["input"] = new OpPort("graph_input", "output");
            }
            else
            {
                ops[i].InputPorts["input"] = new OpPort(ops[i - 1].Id, "output");
            }
        }
        return g;
    }
}
