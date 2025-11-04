# Person Tracking in Multiple Camera Feeds

## Overview

This project implements a robust person tracking system across multiple camera feeds, integrating cutting-edge computer vision models and database management for persistent identity tracking. It combines **YOLOv9** for detection, **DeepSORT** for tracking, and an **OSNet-based ReID** model for cross-camera person re-identification. All identity data is persistently stored in an **SQLite database**.

---

## Key Features

* **Multi-Camera Support:** Seamless tracking of individuals across multiple video feeds concurrently.
* **Person Detection:** Uses **YOLOv9** for precise and fast person localization.
* **Object Tracking:** Employs **DeepSORT** for maintaining consistent track IDs.
* **Re-Identification (ReID):** Integrates **OSNet** for feature extraction and cosine-similarity-based identity matching.
* **Manual Selection:** Enables on-the-fly user intervention for adjusting or locking tracks.
* **Data Persistence:** Uses an **SQLite database** to store and update tracking records.
* **Logging:** Implements detailed event logging using Python’s built-in logging module.

---

## Installation & Setup

### Prerequisites

Ensure the following core libraries are installed:

```bash
pip install ultralytics opencv-python numpy torch torchvision scipy deep-sort-realtime
```

### Deep-Person-ReID Installation

Clone and install the specific version of **Deep-Person-ReID**:

```bash
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -r requirements.txt
python setup.py install
```

---

## Database Initialization

Before running the main script, initialize the SQLite database:

```python
import sqlite3
conn = sqlite3.connect("reid_db.sqlite3")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY, 
        feature BLOB, 
        last_seen_camera INTEGER, 
        last_seen_time REAL
    )
""")
conn.commit()
conn.close()
```

---

## Usage

Run the main execution script by providing paths to your video sources:

```bash
python main.py
```

---

## Configuration Details

| Component       | Configuration Detail                    | Notes                                                 |
| --------------- | --------------------------------------- | ----------------------------------------------------- |
| YOLO Model      | yolov9s.pt                              | Can be changed to other YOLOv9 variants               |
| DeepSORT Params | max_age=30, nn_budget=100               | Tune for balance between stability and re-acquisition |
| ReID Model      | osnet_x1_ain_0.pth                      | Must be present in the working directory              |
| ReID Threshold  | 0.7 (default in `find_matching_person`) | Adjust for stricter or looser identity matching       |

---

## System Workflow

1. **Model Loading:** Initialize and load YOLOv9, DeepSORT, and OSNet ReID models.
2. **Stream Processing:** Each video feed runs in an independent process for parallel tracking.
3. **Detection & Tracking:** YOLO detects persons, DeepSORT assigns temporary IDs, OSNet extracts feature vectors.
4. **Identity Matching:** Extracted features are compared against database entries using cosine similarity.
5. **User Override:** Users can manually correct or lock tracking on specific individuals.
6. **Database Logging:** Updates SQLite with camera ID and timestamp for each person.
7. **Data Output:** Displays final tracked details upon completion.

---

## Example Output

```
[INFO] Processing video: video1.mp4
[INFO] Processing video: video2.mp4
[INFO] ReID Model Loaded Successfully!
[INFO] Selected Person Data: (ID, Feature Vector, Camera, Timestamp)

===== Final Tracked Persons =====
ID | Last Seen Camera | Last Seen Time
---|------------------|---------------------
1  | 0                | 2025-02-18 15:30:45
2  | 1                | 2025-02-18 15:31:12
==========================================
```

---

## Troubleshooting Guide

| Issue                     | Possible Cause                         | Solution                                         |
| ------------------------- | -------------------------------------- | ------------------------------------------------ |
| **CUDA not available**    | PyTorch not installed with GPU support | Reinstall PyTorch with CUDA enabled              |
| **Video not opening**     | Incorrect file path                    | Verify the video source path                     |
| **Low tracking accuracy** | Suboptimal DeepSORT or ReID parameters | Adjust `max_age`, `nn_budget`, or ReID threshold |

---

## Future Improvements

* Develop a **Graphical User Interface (GUI)** for easier manual tracking.
* Migrate to **PostgreSQL** for enhanced scalability.
* Implement **multi-camera synchronization algorithms** for temporal consistency.

---

## Visual References

## Sample Input Video
![Demo](inputSample.gif)

## Sample Output Video
![Demo](outputSample.gif)

---
