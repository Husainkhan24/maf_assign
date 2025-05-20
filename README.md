YouTube Transcript Translation and TTS Webapp
Project Overview
This Flask-based web application processes YouTube videos through three integrated pipelines to extract transcripts, translate them into an Indian language, and convert the translated text into audio. Users can input a YouTube video link via a web interface, and the app delivers the final audio output in the target Indian language.
Pipelines

Pipeline 1: Transcript Extraction
Extracts transcripts from YouTube videos using the provided video link.


Pipeline 2: Translation and Preprocessing
Translates the extracted transcript into an Indian language and preprocesses it using a pretrained model.


Pipeline 3: Text-to-Speech Conversion
Converts the translated text into audio using the BharathTTS model with pretrained weights.



Project Structure
├── main.py                        # Flask app integrating all pipelines
├── pipeline_one_transcript.ipynb  # Pipeline 1: YouTube transcript extraction
├── pipeline2_translation.py       # Pipeline 2: Translation and preprocessing
├── pipeline_3_audio_convert.py    # Pipeline 3: Text-to-speech conversion
├── requirements.txt               # Project dependencies
└── README.md                     # Project documentation

Prerequisites

Python: Version 3.10 or higher
Virtual Environment: Recommended for dependency isolation
Git: For cloning the repository
YouTube Video Links: Must have available transcripts for Pipeline 1

Setup Instructions

Clone the Repository
git clone <repository-url>
cd <repository-directory>


Create and Activate a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies
pip install -r requirements.txt


Run the Flask Application
python main.py


Access the Webapp

Open a browser and navigate to http://localhost:5000.
Enter a YouTube video link in the provided input field to process it through the pipelines.



Pipeline Details
Pipeline 1: Transcript Extraction

Script: pipeline_one_transcript.ipynb
Functionality: Extracts raw transcript text from YouTube videos.
Input: YouTube video URL
Output: Raw transcript text

Pipeline 2: Translation and Preprocessing

Script: pipeline2_translation.py
Functionality: Translates the raw transcript into an Indian language and preprocesses it using a pretrained model.
Input: Raw transcript from Pipeline 1
Output: Translated and preprocessed text

Pipeline 3: Text-to-Speech Conversion

Script: pipeline_3_audio_convert.py
Functionality: Converts the translated text into audio using the BharathTTS model with pretrained weights.
Input: Translated text from Pipeline 2
Output: Audio file in the target Indian language (e.g., assamese_audio.wav)
Output Location: Audio files are saved in the indic-tts/inference directory

Flask Application API

Script: main.py
Functionality: Integrates all three pipelines into a single workflow.
Usage:
Run python main.py to start the Flask server.
Input a YouTube video link via the web interface at http://localhost:5000.
The app processes the link through the pipelines and returns the final audio output.



Notes

Ensure the YouTube video has available transcripts, as Pipeline 1 relies on this.
Pretrained models for translation and BharathTTS must be properly configured and accessible.
Use requirements.txt to avoid dependency conflicts.
For production deployment, consider using a WSGI server like Gunicorn.
Audio outputs from Pipeline 3 are saved as assamese_audio.wav in the indic-tts/inference directory.

