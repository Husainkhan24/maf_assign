from flask import Flask, render_template, request
import re
from urllib.parse import urlparse, parse_qs


from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


import re
from langdetect import detect, LangDetectException




import torch
from IndicTransToolkit.processor import IndicProcessor 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


print("import comp")


app = Flask(__name__)




#####################################################################################


supported_languages = [
    "asm_Beng",  # Assamese (Bengali script)
    "ben_Beng",  # Bengali (Bengali script)
    "brx_Deva",  # Bodo (Devanagari script)
    "doi_Deva",  # Dogri (Devanagari script)
    "eng_Latn",  # English (Latin script)
    "gom_Deva",  # Konkani (Devanagari script)
    "guj_Gujr",  # Gujarati (Gujarati script)
    "hin_Deva",  # Hindi (Devanagari script)
    "kan_Knda",  # Kannada (Kannada script)
    "kas_Arab",  # Kashmiri (Arabic script)
    "kas_Deva",  # Kashmiri (Devanagari script)
    "mai_Deva",  # Maithili (Devanagari script)
    "mal_Mlym",  # Malayalam (Malayalam script)
    "mar_Deva",  # Marathi (Devanagari script)
    "mni_Beng",  # Manipuri (Bengali script)
    "mni_Mtei",  # Manipuri (Meitei script)
    "npi_Deva",  # Nepali (Devanagari script)
    "ory_Orya",  # Odia (Odia script)
    "pan_Guru",  # Punjabi (Gurmukhi script)
    "san_Deva",  # Sanskrit (Devanagari script)
    "sat_Olck",  # Santali (Ol Chiki script)
    "snd_Arab",  # Sindhi (Arabic script)
    "snd_Deva",  # Sindhi (Devanagari script)
    "tam_Taml",  # Tamil (Tamil script)
    "tel_Telu",  # Telugu (Telugu script)
    "urd_Arab"   # Urdu (Arabic script)
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
print("Loading model...")



model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    attn_implementation="flash_attention_2"
).to(DEVICE)


ip = IndicProcessor(inference=True)



#######################################################################################



def get_transcript_languages(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return [t.language_code for t in transcript_list]
    except Exception:
        return None  
    
def get_transcript(video_id, lang):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        return transcript
    except Exception as e:
        # print("Error:", e)
        return []

def is_english(text):
    try:
        language = detect(text)
        return language == 'en'
    except LangDetectException:
        return False  

def clean_transcript_English(text):
    
    words = text.split()
    cleaned_words = [word for word in words if word != '[Music]' and not (word.startswith('[') and word.endswith(']'))]
    # cleaned_words = [word for word in words if not (word.startswith('[') and word.endswith(']'))]
    cleaned_words = [re.sub(r'[^a-zA-Z0-9\s\-\']', '', word) for word in cleaned_words]
    cleaned_words = [word for word in cleaned_words if word]
    if not cleaned_words:
        return None  
    
    return ' '.join(cleaned_words)


def video_transcript_pipeline(vid):

    raw_path = "C:/Users/ROG/Desktop/mafatlaal/web_app/raw_trans"
    proc_path = "C:/Users/ROG/Desktop/mafatlaal/web_app/proc_trans"

    langs = get_transcript_languages(vid)
    if langs:
        english_variants = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU', 'en-IN']
        prime_lang = next((lang for lang in langs if lang in english_variants), langs[0])

        transcript = get_transcript(vid, prime_lang)
        if transcript:
            
            raw_filename = f"{raw_path}/{vid}_{prime_lang}_RAW.txt"
            with open(raw_filename, 'w', encoding='utf-8') as raw_file:
                for segment in transcript:
                    raw_file.write(f"{segment['text']} (Start: {segment['start']}s, Duration: {segment['duration']}s)\n")

            
            proc_filename = f"{proc_path}/{vid}_{prime_lang}_Proces.txt"
            with open(proc_filename, 'w', encoding='utf-8') as proc_file:
                for segment in transcript:
                    text = segment['text']
                    if is_english(text):
                        cleaned = clean_transcript_English(text)       
                    else:
                        cleaned = text  
                    proc_file.write(f"{cleaned}\n")

            print(f"Transcripts saved for video: {vid}")
            return proc_filename,prime_lang
        else:
            print(f"No transcript found for video: {vid}")
            return False,False
    else:
        print(f"No languages found for video: {vid}")
        return False




def extract_youtube_id(url):
    
    parsed_url = urlparse(url)

    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/embed/')[1]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]

    return None






@app.route('/', methods=['GET', 'POST'])
def index():
    transcript_text = ''
    transcript_fail = 'somethiong went wrong'

    if request.method == 'POST':
        user_text = request.form.get('user_input')
        vid_id = extract_youtube_id(user_text)
        res,trigger_lang = video_transcript_pipeline(vid_id)  # Stage one

        if res:
            print("fox fox",trigger_lang)
            print("fox fox",trigger_lang)
            print("fox fox",trigger_lang)
            print("fox fox",trigger_lang)
            print("fox fox",trigger_lang)
            with open(res, 'r', encoding='utf-8') as f:
                transcript_text = f.read()

            
            return render_template('index.html', text=transcript_text)
        else:
            print("awesome worki")
            return render_template('index.html', text=transcript_fail)

    return render_template('index.html')



@app.route('/translate_api', methods=['POST'])
def translate_api():
    data = request.get_json()
    transcript_text = data.get('transcript_text')
    trigger_lang = data.get('trigger_lang')

    print("Translate API Triggered!")
    print("Language:", trigger_lang)
    print("Transcript:", transcript_text[:100])  # Just printing first 100 chars

    return {'status': 'received'}




if __name__ == '__main__':
    app.run(debug=True)
























