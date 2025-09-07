coral_spawn_counter/
├── config/
│   ├── __init__.py
│   └── config_manager.py          # ConfigManager class
├── models/
│   ├── __init__.py
│   └── model_manager.py           # ModelManager class
├── processing/
│   ├── __init__.py
│   ├── image_processor.py         # ImageProcessor class
│   └── batch_processor.py         # BatchProcessor class
├── data/
│   ├── __init__.py
│   ├── detection_data_manager.py  # DetectionDataManager class
│   └── file_manager.py            # FileManager class
├── visualization/
│   ├── __init__.py
│   └── plotter.py                 # DetectionPlotter class
├── utils/
│   ├── __init__.py
│   ├── time_utils.py              # TimeUtils class
│   └── resume_manager.py          # ResumeManager class
└── coral_spawn_predictor.py       # Main CoralSpawnPredictor class (orchestrator)