# Equipment Utilization and Activity Classification

## Overview
This project delivers an end-to-end computer vision pipeline for construction-site equipment intelligence. It combines detection, tracking, motion analytics, and activity inference to produce operational utilization metrics that are ready for reporting and decision support.

The solution is built for practical field scenarios and produces both visual and analytical outputs in a single workflow.

## Projects
- Baseline Video Intelligence Pipeline: [script/pipeline.py](script/pipeline.py)
  - Detects and tracks heavy equipment from video streams.
  - Infers ACTIVE and INACTIVE state in real time.
  - Classifies operational activities such as DIGGING, SWINGING_LOADING, DUMPING, MOVING, and WAITING.
  - Computes utilization KPIs per equipment ID.
- Colab-Ready End-to-End Notebook: [script/Equipment Utilization & Activity Classification.ipynb](script/Equipment%20Utilization%20%26%20Activity%20Classification.ipynb)
  - Includes ACID dataset fine-tuning workflow.
  - Runs full pipeline execution and output inspection.
  - Builds analytics database and dashboard-ready tables.
  - Generates deployment bundle artifacts.

## Repository Structure
```text
.
|-- outputs/
|   |-- equipment_utilization_delivery.zip
|   |-- output sample.mp4
|-- script/
|   |-- Equipment Utilization & Activity Classification.ipynb
|   |-- pipeline.py
|-- README.md
```

## Key Features
- End-to-end workflow from raw video to utilization analytics.
- Robust ID continuity logic with reassociation gates and overlap-aware handling.
- Motion-aware ACTIVE/INACTIVE detection with smoothing and hold policies.
- Class-aware activity mapping for construction equipment categories.
- Multi-format outputs for engineering, analytics, and stakeholder communication.
- Colab-compatible notebook flow for rapid experimentation and reproducibility.
- Delivery-oriented packaging with deployable bundles and analytics tables.

## Requirements
- Python 3.10+
- CUDA-enabled GPU recommended for faster inference and training

Core Python packages:
- ultralytics
- opencv-python
- numpy
- pandas
- tqdm
- torch

Install example:
```bash
pip install ultralytics opencv-python numpy pandas tqdm torch
```

## Setup
1. Clone the repository.
2. Open the project in VS Code, Jupyter, or Google Colab.
3. Place your source video where your runtime expects it.
4. Run either:
   - The notebook workflow in [script/Equipment Utilization & Activity Classification.ipynb](script/Equipment%20Utilization%20%26%20Activity%20Classification.ipynb), or
   - The script workflow in [script/pipeline.py](script/pipeline.py).

## Usage
Example script run:
```bash
python script/pipeline.py \
  --input_video "/path/to/input.mp4" \
  --output_dir "./outputs/baseline_foundation" \
  --weights "yolov8x.pt"
```

The notebook provides an extended production-style workflow including fine-tuning, validation, analytics, QA reporting, and deploy bundle generation.

## Output
Typical generated artifacts:
- Annotated video with tracking overlays and utilization state
- Frame-level events in JSONL format
- Per-equipment summary CSV
- SQLite analytics database and dashboard tables

Sample output video:

<video src="outputs/output%20sample.mp4" controls width="100%"></video>

Sample output: [outputs/output sample.mp4](outputs/output%20sample.mp4)

## Notes
- The pipeline is designed for construction equipment operations and utilization analysis.
- The notebook includes operational steps for data prep, model training, run-time tuning, analytics extraction, and delivery packaging.

## Contributing
Contributions are welcome through pull requests. For major updates, please open an issue first to discuss scope and approach.

## License
This project is available for educational and research use. You can add a dedicated license file if required by your organization.

## Author
- GitHub: [samir-m0hamed](https://github.com/samir-m0hamed)
