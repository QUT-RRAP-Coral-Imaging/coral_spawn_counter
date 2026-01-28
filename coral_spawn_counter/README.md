# Coral Spawn Counter Module

This folder contains the core functionality for the Coral Spawn Counter system, which provides automated detection and analysis of coral spawn/larvae in underwater images captured by C-SLIC devices.

## Overview

The module consists of two main components:

1. **[`coral_spawn_predictor.py`](coral_spawn_predictor.py)** - Runs AI model predictions on images to detect coral spawn
2. **[`coral_spawn_counter.py`](coral_spawn_counter.py)** - Analyzes detection results and compares them with manual counts

## Main Components

### CoralSpawnPredictor

The `CoralSpawnPredictor` class is the main orchestrator for running AI model predictions on coral spawn images.

**Key Features:**
- Supports both surface and subsurface detection models
- Batch processing for improved efficiency
- Resume capability for interrupted processing
- Parallel processing support
- Automatic plot generation

**Usage:**
```python
from coral_spawn_counter.coral_spawn_predictor import CoralSpawnPredictor

# Initialize with config file
predictor = CoralSpawnPredictor(config_file)

# Run predictions
predictor.run()

# Generate plots
predictor.plot_surface_detections()
predictor.plot_subsurface_detections()
```

### CoralSpawnCounter

The `CoralSpawnCounter` class handles analysis of detection results and comparison with manual counts.

**Key Features:**
- Processes manual count data from Excel files
- Analyzes surface and subsurface detection results
- Generates comparative plots and statistics
- Exports count data to JSON format
- Tank-level count estimation with scaling

**Usage:**
```python
from coral_spawn_counter.coral_spawn_counter import CoralSpawnCounter

# Initialize with config file
counter = CoralSpawnCounter(config_path)

# Run full analysis
results = counter.run_full_analysis(
    show_plots=False,
    include_surface=True,
    include_subsurface=True,
    export_counts=True
)
```

## Directory Structure

```
coral_spawn_counter/
├── config/                     # Configuration management
│   ├── config_manager.py       # Prediction config management
│   └── plot_config_manager.py  # Analysis config management
├── models/                     # AI model management
│   └── model_manager.py        # YOLO model loading and inference
├── processing/                 # Data processing utilities
│   ├── batch_processor.py      # Batch processing for predictions
│   ├── image_processor.py      # Individual image processing
│   └── invalid_times_processor.py  # Invalid time range handling
├── data/                       # Data management
│   ├── detection_data_manager.py   # Detection data handling
│   └── file_manager.py         # File system operations
├── manual_counts/              # Manual count processing
│   └── manual_counts_processor.py  # Excel file processing
├── visualisation/              # Plotting and visualization
│   ├── detection_plotter.py    # Detection result plots
│   └── tank_count_plotter.py   # Tank count comparison plots
├── utils/                      # Utility functions
│   ├── time_utils.py           # Time/datetime utilities
│   └── resume_manager.py       # Processing resume functionality
└── deprecated_scripts/         # Legacy code (not actively maintained)
```

## Configuration Files

Both main classes use JSON configuration files:

- **Prediction configs** (for `CoralSpawnPredictor`): Located in `data_yaml_files/prediction`
- **Analysis configs** (for `CoralSpawnCounter`): Located in `data_yaml_files/plotting`

## Output Structure

The system creates organized output directories:

```
base_detection_dir/
└── cslics_uuid/
    ├── plots/              # Generated plots and visualizations
    │   └── batch_histogram/ # Batch-level histogram plots
    ├── data/               # CSV exports and processed data
    ├── surface/            # Surface detection files (.txt)
    └── subsurface/         # Subsurface detection files (.txt)
```

## Key Features

### Detection Pipeline
1. **Image Processing**: Batch processing of C-SLIC images
2. **AI Inference**: YOLO-based object detection for coral spawn
3. **Data Export**: Structured output in multiple formats (JSON, CSV, plots)

### Analysis Pipeline
1. **Manual Count Integration**: Processes Excel-based manual counts
2. **Detection Analysis**: Analyzes AI detection results over time
3. **Comparative Analysis**: Compares automated vs manual counts
4. **Tank Estimation**: Scales image-level counts to tank-level estimates

### Export Capabilities
- JSON export with metadata (`export_counts_to_json`)
- CSV data exports for further analysis
- High-quality plots for publications and reports

## Example Usage

See `example_export_counts.py` for comprehensive usage examples, including:
- Automatic export during analysis
- Manual export from results
- Individual dataset exports

## Related Files

- `example_export_counts.py` - Usage examples
- `EXPORT_COUNTS_README.md` - Export functionality documentation
- `setup.py` - Package installation script
