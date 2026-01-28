# Export Counts to JSON Documentation

## Overview

The `CoralSpawnCounter` class now includes functionality to export manual counts, subsurface counts, and surface counts data to JSON files. This allows users to recreate the plots produced by the surface and subsurface analysis.

## New Methods

### `export_counts_to_json()`

```python
def export_counts_to_json(self, manual_counts, subsurface_counts=None, surface_counts=None, 
                         output_path=None, include_metadata=True):
```

**Purpose**: Export count data to a structured JSON file that can be used to recreate analysis plots.

**Parameters**:
- `manual_counts`: Manual count data (dict or array-like)
- `subsurface_counts`: Subsurface detection data (dict or DataFrame, optional)
- `surface_counts`: Surface detection data (dict or DataFrame, optional)
- `output_path`: Path to save JSON file. If None, uses default location.
- `include_metadata`: Whether to include configuration metadata

**Returns**: Path to the saved JSON file

### `export_analysis_results()`

```python
def export_analysis_results(self, results, output_path=None, include_metadata=True):
```

**Purpose**: Convenience method to export results from `run_full_analysis()`.

**Parameters**:
- `results`: Results dictionary from `run_full_analysis()`
- `output_path`: Path to save JSON file. If None, uses default location.
- `include_metadata`: Whether to include configuration metadata

**Returns**: Path to the saved JSON file

### Updated `run_full_analysis()`

The `run_full_analysis()` method now includes an optional `export_counts` parameter:

```python
def run_full_analysis(self, show_plots=False, include_surface=True, 
                     include_subsurface=True, export_counts=True):
```

When `export_counts=True` (default), the method automatically exports all processed data to JSON.

## JSON File Structure

The exported JSON file contains the following structure:

```json
{
  "export_timestamp": "2025-09-10T14:30:00.123456",
  "data_version": "1.0",
  "metadata": {
    "cslics_uuid": "cslics08",
    "coral_species": "Acropora loripes",
    "tank_sheet_name": "tank4",
    "confidence_threshold": 0.5,
    "submersion_time": "2023-12-05T18:00:00",
    "surface_model_name": "yolo_surface_model",
    "subsurface_model_name": "yolo_subsurface_model",
    "surface_calibration": {
      "idx": 10,
      "window_size": 5,
      "window_shift": 1
    },
    "subsurface_calibration": {
      "idx": 15,
      "window_size": 3,
      "window_shift": 1
    }
  },
  "manual_counts": {
    "counts": [100, 150, 200, ...],
    "std": [10, 15, 20, ...],
    "counts_time": ["2023-12-05T18:30:00", "2023-12-05T19:00:00", ...],
    "decimal_days": [0.5, 1.0, 1.5, ...],
    "camera_uuid": "camera123",
    "species": "Acropora loripes",
    "nearest_day": "2023-12-06T00:00:00"
  },
  "subsurface_counts": {
    "image_counts": [80, 120, 180, ...],
    "image_times": ["2023-12-05T18:30:00", "2023-12-05T19:00:00", ...],
    "decimal_days": [0.5, 1.0, 1.5, ...],
    "total_counts_scaled": [85, 125, 185, ...],
    "scale_factor": 1.0625,
    "errors": []
  },
  "surface_counts": {
    "timestamp": ["2023-12-05T18:30:00", "2023-12-05T19:00:00", ...],
    "hours_since_spawning": [0.5, 1.0, 1.5, ...],
    "class_0": [10, 15, 20, ...],
    "class_1": [5, 8, 12, ...],
    "spawn": [50, 75, 100, ...]
  }
}
```

## Usage Examples

### 1. Automatic Export During Analysis

```python
from coral_spawn_counter.coral_spawn_counter import CoralSpawnCounter

config_path = "path/to/your/config.json"
counter = CoralSpawnCounter(config_path)

# Run analysis with automatic export
results = counter.run_full_analysis(
    show_plots=False,
    include_surface=True,
    include_subsurface=True,
    export_counts=True  # Automatically exports data
)
```

### 2. Manual Export from Results

```python
# Run analysis without automatic export
results = counter.run_full_analysis(export_counts=False)

# Export manually
export_path = counter.export_analysis_results(results)
print(f"Data exported to: {export_path}")

# Export to custom location
custom_path = counter.export_analysis_results(
    results, 
    output_path="/custom/path/export.json"
)
```

### 3. Export Individual Datasets

```python
# Process only manual counts
manual_data = counter.process_manual_counts()

# Export only manual counts
counter.export_counts_to_json(
    manual_counts=manual_data,
    output_path="/path/to/manual_only.json"
)

# Process surface data and export with manual data
surface_data = counter.process_surface_detections()
counter.export_counts_to_json(
    manual_counts=manual_data,
    surface_counts=surface_data,
    output_path="/path/to/surface_and_manual.json"
)
```

## File Locations

By default, exported JSON files are saved to:
```
{base_detection_dir}/{cslics_uuid}/data/counts_export_{tank_sheet_name}_{cslics_uuid}.json
```

For example:
```
/path/to/detections/cslics08/data/counts_export_tank4_cslics08.json
```

## Data Types and Formats

- **Timestamps**: Converted to ISO format strings (e.g., "2023-12-05T18:30:00")
- **Decimal Days**: Hours since spawning divided by 24
- **Counts**: Integer arrays representing detection/manual counts
- **Scale Factors**: Float values used for calibration
- **Class Data**: Each detection class becomes a separate array in surface counts

## Error Handling

The export methods include comprehensive error handling:
- Invalid data types are handled gracefully
- Missing optional data is skipped without errors
- File system errors are caught and reported
- Progress and summary information is printed during export

## Recreating Plots

The exported JSON contains all necessary data to recreate the original plots:

1. **Manual Counts Plot**: Use `manual_counts.counts` vs `manual_counts.decimal_days`
2. **Surface Detection Plot**: Use `surface_counts.{class_name}` vs `surface_counts.hours_since_spawning`
3. **Subsurface Detection Plot**: Use `subsurface_counts.total_counts_scaled` vs `subsurface_counts.decimal_days`
4. **Calibration**: Apply scale factors and calibration parameters from metadata

## Notes

- All datetime objects are converted to ISO format strings for JSON compatibility
- The export preserves the complete data structure needed for analysis recreation
- Metadata includes all configuration parameters used in the original analysis
- The JSON format is human-readable and can be easily parsed by other tools
