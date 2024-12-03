# Audio to Image Converter

This project converts audio files to video by generating images based on the audio content using AI.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-to-image.git
cd audio-to-image
```

2. Create and activate virtual environment:

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example configuration file:
```bash
cp config.example.py config.py
```

2. Edit `config.py` and replace `your-key` with your Silicon Flow API token:
```python
SILICON_FLOW_API_TOKEN = "your-actual-token"
```

## Usage

Run the script with your audio file:
```bash
python audio_to_video.py --input your_audio.wav
```

The script will:
1. Process the audio file
2. Generate images using AI
3. Create a video combining the audio and generated images
4. Save the output as `output.mp4`

## Project Structure

- `audio_to_video.py`: Main script for audio processing and video generation
- `config.py`: Configuration file for API tokens
- `requirements.txt`: Project dependencies

## Dependencies

- requests>=2.31.0: For API communication
- moviepy>=1.0.3: For video creation
- colorlog: For colored logging output
- tqdm: For progress bars

## Output

- Generated images will be saved as PNG files
- Final video will be saved as `output.mp4`