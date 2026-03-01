using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace NeuralSim.App.Views;

/// <summary>
/// Converts non-empty string to Visible, empty/null to Collapsed.
/// Used inline in XAML for showing tensor values only when present.
/// </summary>
public class StringToVisConverter : IValueConverter
{
    public static readonly StringToVisConverter Instance = new();

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        // Support both string (empty → collapsed) and int (0 → collapsed)
        if (value is int count)
            return count > 0 ? Visibility.Visible : Visibility.Collapsed;
        return string.IsNullOrWhiteSpace(value?.ToString()) ? Visibility.Collapsed : Visibility.Visible;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
