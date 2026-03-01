using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace NeuralSim.App.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    /// <summary>Start drag from palette item.</summary>
    private void PaletteItem_MouseMove(object sender, MouseEventArgs e)
    {
        if (e.LeftButton != MouseButtonState.Pressed) return;
        if (sender is not Border border || border.Tag is not string opType) return;

        var data = new DataObject();
        data.SetData("OpType", opType);
        DragDrop.DoDragDrop(border, data, DragDropEffects.Copy);
    }
}
