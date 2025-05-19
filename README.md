YouTube Transcript Translation and TTS Webapp
Project Overview
This project is a Flask-based web application that integrates three pipelines to process YouTube videos:

Pipeline 1: Extracts transcripts from YouTube videos.
Pipeline 2: Translates the extracted transcript into an Indian language and preprocesses it using a pretrained model.
Pipeline 3: Converts the translated transcript into audio in the target Indian language using the BharathTTS model with pretrained weights.

The webapp allows users to input a YouTube link, processes it through the three pipelines, and delivers the final audio output.
Project Structure
├── main.py                      # Flask app integrating all pipelines
├── pipeline_one_transcript.ipynb # Pipeline 1: YouTube transcript extraction
├── pipeline2_translation.py      # Pipeline 2: Translation and preprocessing
├── pipeline_3_audio_convert.py   # Pipeline 3: Text-to-speech conversion
├── requirements.txt            
└── README.md                    

Prerequisites

Python 3.10+
Virtual environment (recommended)
Git
YouTube video links with available transcripts

Setup Instructions

Clone the Repository:
git clone <your-repository-url>
cd <repository-name>


Create and Activate a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Run the Flask Application:
python main.py


Access the Webapp:

Open a browser and navigate to http://localhost:5000.
Enter a YouTube video link in the provided input field to process it through the pipelines.



Pipeline Details
Pipeline 1: Transcript Extraction

Script: pipeline_one_transcript.ipynb
Functionality: Extracts transcripts from YouTube videos using the provided video link.
Input: YouTube video URL.
Output: Raw transcript text.

Pipeline 2: Translation and Preprocessing

Script: pipeline2_translation.py
Functionality: Translates the transcript into an Indian language and preprocesses it using a pretrained model.
Input: Raw transcript from Pipeline 1.
Output: Translated and preprocessed text.

Pipeline 3: Text-to-Speech Conversion

Script: pipeline_3_audio_convert.py
Functionality: Converts the translated text into audio using the BharathTTS model with pretrained weights.
Input: Translated text from Pipeline 2.
Output: Audio file in the target Indian language.

Flask Application API

Script: main.py
Functionality: Integrates all three pipelines into a single workflow.
Usage:
Run main.py to start the Flask server.
Input a YouTube link via the web interface.
The app processes the link through the pipelines and returns the final audio output.



Notes

Ensure the YouTube video has transcripts available, as Pipeline 1 relies on this.
The pretrained models for translation and TTS (BharathTTS) must be properly configured and accessible.
Use the custom_equirements.txt to avoid dependency conflicts.
For production, consider deploying with a WSGI server like Gunicorn.




