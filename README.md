# Fly Behavior Analysis

## Overview

This project aims to analyze the behavior of flies in a video. It specifically focuses on tracking Regions of Interest (ROIs), the movement of individual flies within these regions, and detects mating events based on video data. The application is built using PyQt6 for the graphical user interface and OpenCV for video processing.

## Features

- **Select Video**: Allows the user to select a video file for analysis.
- **Start/Stop Processing**: Provides buttons to start and stop video analysis.
- **Frame and Time Information**: Displays the current frame number and time in the video.
- **Mating Event Detection**: Detects and tracks mating events, providing both the start time and duration.
- **Binary Image Display**: Shows a thresholded (binary) version of the video for debugging purposes.
- **Fly Center Mating Event Detection**:  Detects and tracks mating events in the center of ROI
- **Fly Information**: Displays detailed information about each detected fly.
- **Fly center Info**: Displays detailed information about each detected fly.
- **Export DataFrame**: Allows the user to export data into a CSV file.

## Dependencies

- Python 3.x
- PyQt6
- OpenCV (cv2)
- NumPy
- Pandas

## Installation

First, clone the repository:

```bash
git clone https://github.com/SomeOne1Random/drosophila-melanogaster-mating-time-finder-python
```

Then navigate to the project directory and install the required packages:

```bash
cd fly-behavior-analysis
pip install -r requirements.txt
```

## Usage

Run the main Python script:

```bash
python main.py
```

This will open the application where you can select a video and start the analysis.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License. See `LICENSE` for more information.

---

Feel free to adapt this README to better fit your project's specific needs!
