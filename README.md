# Multi-View Foul Recognition

Sick of refs getting it wrong?

Introducing Multi-View Foul Recognition, the AI-powered solution designed to make football refereeing more accurate and fair. By leveraging multiple camera angles, this system can detect fouls with precision and assist referees in making real-time decisions based on comprehensive video analysis. Say goodbye to missed calls and controversial decisions!

## 🚀 Features

* Multi-Angle Video Analysis: Analyzes footage from multiple camera angles to accurately detect fouls in football matches.
* AI-Powered Recognition: Uses machine learning to distinguish between different types of fouls, ensuring consistent and fair officiating.
* Real-Time Decision Support: Provides immediate feedback to referees, reducing human error and enhancing match integrity.
* Scalable: Easily integrates with existing camera setups to process live footage or pre-recorded matches.
* Open Source: Free for anyone to use, modify, and contribute to the development of more accurate foul detection systems.

## 📚 Dataset

This project uses the SoccerNet-MVFoul dataset, which can be downloaded via API:

```bash
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="path/to/SoccerNet")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password="enter password")
```

The incident tends to occur close to frame 75 (3 seconds) so previous studies have taken frames 63 to 87 as input.

## 🔧 Installation

### Prerequisites
* Python 3.7 or higher
* TensorFlow or PyTorch (depending on the model you're using)
* OpenCV for video processing
* FFmpeg (for video conversion and frame extraction)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/ghchen99/multi-view-foul-recognition.git
cd multi-view-foul-recognition
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Download the pre-trained models from the release page or train your own model by following the instructions in the training folder.
4. To run the demo, use the following command:
```bash
python main.py --input_video your_video.mp4
```

## ⚙️ How It Works

This project uses advanced machine learning techniques to detect fouls in football matches through multi-angle video footage. Here's a breakdown of how the system operates:

* Input Video: The system takes in a football match video captured from multiple camera angles.
* Video Processing: Using OpenCV, the video is processed to extract frames from different angles.
* AI Analysis: The extracted frames are analyzed using a pre-trained deep learning model, which has been trained to detect common football fouls (like tackles, handballs, or off-the-ball incidents).
* Decision Support: The system identifies potential fouls and marks them, providing timestamps and a confidence score for each detection.
* Referee Assistance: The system can provide suggestions to the referee, who can review the identified foul instances for a more informed decision.

## 📂 Folder Structure

```bash
/multi-view-foul-recognition-2025
│
├── /src               # Source code for video processing and model inference
├── /training          # Scripts and resources for training the AI model
├── /data              # Folder for storing dataset and pre-processed videos
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── main.py            # Main entry point for running the system
```

## 🤝 Contributing

We welcome contributions from the community! If you have ideas to improve the system or fix bugs, feel free to fork the repository and submit a pull request.

### Steps to Contribute:

1. Fork the repository.
2. Create a new branch `git checkout -b feature-branch`.
3. Make your changes.
4. Commit your changes `git commit -am 'Add new feature'`.
5. Push to your branch `git push origin feature-branch`.
6. Create a new pull request.


