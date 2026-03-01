using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using NeuralSim.App.ViewModels;

namespace NeuralSim.App.Views;

public partial class GraphCanvas : UserControl
{
    // Node dragging state
    private CanvasNodeViewModel? _draggingNode;
    private Point _dragStart;
    private double _dragNodeStartX, _dragNodeStartY;
    private bool _isDragging;

    private MainViewModel VM => (MainViewModel)DataContext;

    public GraphCanvas()
    {
        InitializeComponent();
        KeyDown += OnKeyDown;
        Focusable = true;
    }

    // ───── Helpers ─────

    /// <summary>Walk up the visual tree to check if any ancestor is of type T.</summary>
    private static bool IsChildOf<T>(DependencyObject? child, DependencyObject stop) where T : DependencyObject
    {
        while (child != null && child != stop)
        {
            if (child is T) return true;
            child = VisualTreeHelper.GetParent(child);
        }
        return false;
    }

    // ───── Drag-drop from palette ─────

    private void OnDragOver(object sender, DragEventArgs e)
    {
        if (e.Data.GetDataPresent("OpType"))
            e.Effects = DragDropEffects.Copy;
        else
            e.Effects = DragDropEffects.None;
        e.Handled = true;
    }

    private void OnDrop(object sender, DragEventArgs e)
    {
        if (!e.Data.GetDataPresent("OpType")) return;
        var opType = (string)e.Data.GetData("OpType");
        var pos = e.GetPosition(this);

        // Snap to reasonable position (center the node)
        double x = Math.Max(0, pos.X - CanvasNodeViewModel.NodeWidth / 2);
        double y = Math.Max(0, pos.Y - CanvasNodeViewModel.NodeHeight / 2);

        VM.AddNodeToCanvasCommand.Execute(new PaletteDropInfo(opType, x, y));
        e.Handled = true;
    }

    // ───── Node dragging ─────

    private void Node_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (sender is not FrameworkElement fe || fe.Tag is not CanvasNodeViewModel node) return;

        // Don't start drag if clicking inside a TextBox (editable input node)
        if (e.OriginalSource is DependencyObject src && IsChildOf<TextBox>(src, fe)) return;

        // Select the node
        VM.SelectNodeCommand.Execute(node);
        Focus(); // Ensure keyboard focus so DEL key works

        // Start drag
        _draggingNode = node;
        _dragStart = e.GetPosition(this);
        _dragNodeStartX = node.X;
        _dragNodeStartY = node.Y;
        _isDragging = false;
        fe.CaptureMouse();
        e.Handled = true;
    }

    private void Node_MouseMove(object sender, MouseEventArgs e)
    {
        if (_draggingNode == null || e.LeftButton != MouseButtonState.Pressed) return;
        var pos = e.GetPosition(this);
        double dx = pos.X - _dragStart.X;
        double dy = pos.Y - _dragStart.Y;

        if (!_isDragging && (Math.Abs(dx) > 3 || Math.Abs(dy) > 3))
            _isDragging = true;

        if (_isDragging)
        {
            _draggingNode.X = Math.Max(0, _dragNodeStartX + dx);
            _draggingNode.Y = Math.Max(0, _dragNodeStartY + dy);
            VM.OnNodeMoved(_draggingNode);
        }
    }

    private void Node_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (sender is FrameworkElement fe)
            fe.ReleaseMouseCapture();
        _draggingNode = null;
        _isDragging = false;
    }

    // ───── Port clicking (wiring) ─────

    private void OutputPort_Click(object sender, MouseButtonEventArgs e)
    {
        if (sender is not FrameworkElement fe || fe.Tag is not CanvasNodeViewModel node) return;
        e.Handled = true;

        var info = new PortClickInfo(node, "output", IsOutput: true);
        VM.PortClickedCommand.Execute(info);
        UpdateWiringLabel();
        Focus(); // so ESC works
    }

    private void InputPort_Click(object sender, MouseButtonEventArgs e)
    {
        if (sender is not FrameworkElement fe || fe.Tag is not CanvasNodeViewModel node) return;
        e.Handled = true;

        // For simplicity, use first input port name
        string portName = node.InputPortNames.Count > 0 ? node.InputPortNames[0] : "input";
        var info = new PortClickInfo(node, portName, IsOutput: false);
        VM.PortClickedCommand.Execute(info);
        UpdateWiringLabel();
        Focus();
    }

    private void OnKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Escape && VM.IsWiring)
        {
            VM.CancelWiringCommand.Execute(null);
            UpdateWiringLabel();
        }
        else if (e.Key == Key.Delete && VM.SelectedNode != null && !VM.SelectedNode.IsFixed)
        {
            VM.RemoveNodeCommand.Execute(VM.SelectedNode);
        }
    }

    private void UpdateWiringLabel()
    {
        WiringLabel.Visibility = VM.IsWiring ? Visibility.Visible : Visibility.Collapsed;
    }
}
