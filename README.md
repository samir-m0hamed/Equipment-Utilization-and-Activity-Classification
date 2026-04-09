# Equipment Utilization and Activity Classification

## Overview
This project presents a production-oriented computer vision system for analyzing construction equipment operations. It combines real-time detection, tracking, motion analysis, and activity classification within a scalable microservices architecture.

The system is designed to transform raw video streams into actionable utilization insights, enabling monitoring of equipment efficiency and operational behavior.

---

## Live Demo

An interactive deployment is available on Hugging Face Spaces, showcasing real-time equipment tracking, activity classification, and utilization analytics:

[Live Demo on Hugging Face](https://huggingface.co/spaces/samir-mohamed/Equipment-Utilization-Activity-Classifier)

---

## Project Objective
The goal of this project is to design and implement a **real-time, distributed pipeline** that:

- Processes video streams of construction equipment  
- Tracks equipment utilization states **(ACTIVE vs INACTIVE)**  
- Classifies operational activities such as **DIGGING, SWINGING_LOADING, DUMPING, MOVING, and WAITING.** 
- Computes working vs dwell time  
- Streams results through a message broker to a user interface  

---

## System Architecture

The solution follows a **microservices-based architecture** with asynchronous communication using Apache Kafka.

### Core Components

#### 1. Computer Vision Service
- Detects and tracks equipment (e.g., excavators, dump trucks)
- Maintains consistent object IDs across frames
- Performs motion-aware state classification:
  - ACTIVE (moving / operating)
  - INACTIVE (stationary)

#### 2. Motion Intelligence Module
- Handles articulated motion scenarios (e.g., excavator arm moving while base is static)
- Uses region-aware motion analysis and temporal smoothing
- Ensures accurate state detection even under partial movement

#### 3. Activity Classification Module
Classifies fine-grained activities:
- DIGGING  
- SWINGING / LOADING  
- DUMPING  
- MOVING  
- WAITING  

#### 4. Streaming Layer (Apache Kafka)
- Streams real-time inference results between services
- Decouples processing pipeline for scalability and fault tolerance

#### 5. Data Sink
- Stores processed analytics in:
  - PostgreSQL / TimescaleDB
- Enables historical analysis and dashboard integration

#### 6. Analytics & UI Layer
- Provides a lightweight interface (Streamlit / Gradio)
- Displays:
  - Processed video with bounding boxes
  - Live equipment status (ACTIVE / INACTIVE)
  - Current activity per machine
  - Utilization dashboard:
    - Total Working Time
    - Total Idle Time
    - Utilization Percentage

---

## Repository Structure
~~~text
.
|-- outputs/
|   |-- equipment_utilization_delivery.zip
|   |-- output sample.mp4
|-- script/
|   |-- Equipment Utilization & Activity Classification.ipynb
|   |-- pipeline.py
|-- README.md
~~~

---

## Key Features
- Real-time video processing pipeline for construction environments  
- Motion-aware utilization tracking with temporal smoothing  
- Robust handling of articulated machinery motion  
- Fine-grained activity classification  
- Distributed architecture using Apache Kafka  
- Persistent analytics storage (PostgreSQL / TimescaleDB)  
- Interactive visualization through a lightweight UI  
- End-to-end workflow from inference to analytics  

---

## Implemented Capabilities (Requirement Coverage)

### Equipment Utilization Tracking
- Real-time classification of equipment into ACTIVE / INACTIVE states  
- Motion-based inference with stability controls  

### Handling Articulated Motion
- Detects activity even when only parts of equipment are moving  
- Uses region-based motion logic and temporal consistency  

### Activity Classification
- Supports multiple construction-specific activities  
- Produces frame-level and aggregated predictions  

### Working vs Idle Time Calculation
- Computes:
  - Total Active Time  
  - Total Idle Time  
  - Utilization Percentage = Active Time / Total Time  

### Data Streaming
- Real-time event streaming using Apache Kafka  
- Structured messages for downstream processing  

### Data Persistence
- Stores results in analytical databases  
- Enables querying, reporting, and dashboarding  

### User Interface
- Live annotated video feed  
- Real-time machine status  
- Utilization metrics dashboard  

---

## Requirements
- Python 3.10+
- CUDA-enabled GPU (recommended)

### Core Dependencies
- ultralytics  
- opencv-python  
- numpy  
- pandas  
- tqdm  
- torch  

Install:
~~~bash
pip install ultralytics opencv-python numpy pandas tqdm torch
~~~

---

## Setup
1. Clone the repository  
2. Open using VS Code, Jupyter, or Google Colab  
3. Provide input video  

### Run (Script Mode)
~~~bash
python script/pipeline.py \
  --input_video "/path/to/input.mp4" \
  --output_dir "./outputs/baseline_foundation" \
  --weights "best.pt"
~~~

### Run (Notebook Mode)
Open and execute:
~~~text
script/Equipment Utilization & Activity Classification.ipynb
~~~

---

## Output

Generated artifacts include:
- Annotated video with tracking overlays  
- Frame-level events (JSONL)  
- Per-equipment summaries (CSV)  
- Analytics database (SQLite / PostgreSQL)  

### Sample Output
<video src="outputs/output%20sample.mp4" controls width="100%"></video>

[Download sample video](outputs/output%20sample.mp4)

---

## Notes
- Designed specifically for construction equipment analytics  
- Can be extended to other industrial monitoring scenarios  
- Architecture supports scaling and real-time deployment  

---

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

---

## Author
Samir Mohamed Samir  
GitHub: https://github.com/samir-m0hamed
