using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using NeuralSim.App.ViewModels;
using NeuralSim.Core;

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

    private void SaveProject_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var vm = (MainViewModel)DataContext;
            var project = vm.ExportProject();
            var json = project.ToJson();

            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "NeuralSim Project (*.nsim)|*.nsim",
                DefaultExt = ".nsim",
                FileName = "model"
            };

            if (dialog.ShowDialog() == true)
            {
                File.WriteAllText(dialog.FileName, json);
                vm.StepInfo = $"Saved → {Path.GetFileName(dialog.FileName)}";
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Save failed: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void OpenProject_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "NeuralSim Project (*.nsim)|*.nsim|All Files (*.*)|*.*",
            };

            if (dialog.ShowDialog() == true)
            {
                var json = File.ReadAllText(dialog.FileName);
                var project = ProjectFile.FromJson(json);
                var vm = (MainViewModel)DataContext;
                vm.ImportProject(project);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Open failed: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }
}
