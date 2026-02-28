using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace NeuralSim.App.Converters;

/// <summary>
/// Converts op node status ("idle", "active", "done") to a brush for the node background.
/// </summary>
public class StatusToBrushConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        return value?.ToString() switch
        {
            "active" => (Brush)Application.Current.FindResource("NodeActiveBrush"),
            "done" => (Brush)Application.Current.FindResource("NodeDoneBrush"),
            _ => (Brush)Application.Current.FindResource("NodeDefaultBrush"),
        };
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}

/// <summary>
/// Bool to Visibility converter.
/// </summary>
public class BoolToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        => value is true ? Visibility.Visible : Visibility.Collapsed;

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
