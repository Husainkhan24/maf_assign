import io
from TTS.utils.synthesizer import Synthesizer
from scipy.io.wavfile import write as scipy_wav_write
from src.inference import TextToSpeechEngine


DEFAULT_SAMPLING_RATE = 22050


lan_weight = "C:/Users/ROG/Downloads/as"
lang = "as"


try:
    synthesizer = Synthesizer(
        tts_checkpoint=f"{lan_weight}/as/fastpitch/best_model.pth",
        tts_config_path=f"{lan_weight}/as/fastpitch/config.json",
        tts_speakers_file=f"{lan_weight}/as/fastpitch/speakers.pth",
        tts_languages_file=None,
        vocoder_checkpoint=f"{lan_weight}/as/hifigan/best_model.pth",
        vocoder_config=f"{lan_weight}/as/hifigan/config.json",
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=False,
    )
    print("Assamese synthesizer initialized successfully.")
except Exception as e:
    print(f"Error initializing synthesizer: {e}")
    exit(1)


models = {
    "as": synthesizer,

}
engine = TextToSpeechEngine(models)




# text = "নমস্কাৰ এইটো এটা পৰীক্ষামূলক বাক্য।"

# text ="জীয়া জীয়া যেন মৰমেৰে কোনে ৰিঙিয়াই আজি কাষলে মাতে সাতোৰঙি ৰঙৰ পোহৰৰ চৌদিখে আজি পাহি মেলি ধীৰে ধীৰে বৈ আছে বুকুত কিদৰে ম‍ই সজাও সচাকৈ এটি মিঠা সুৰেৰে মোৰ মন ল’লা কলিজা মোৰ সচাকৈ আজি জোনাকি ফাগুনৰ আবেলি নীলিম আকাশ আজি ৰ’ৱ থমকি তুমি হ’বানে মোৰ প্ৰতি দিশে দিশে আজি নীয়ৰে সজালে ধৰণী তুমি নাজাবা আজি মোৰ পৰা গ’লে মনে উচুপিব ফেকুৰি পলে পলে বিচাৰে আজি দুহাতে মন দুপাখি মেলি  তুমিনো কোন সৰগৰ পৰী"

text = "এই চহৰ আজিও আপোন অতি সুখে দুখে তুমিয়েই মোৰ যেন লগৰী বুকুতে ৰাখো স্মৃতি সোণোৱালী ও গুৱাহাটী এই চহৰ আজিও আপোন অতি সুখে দুখে তুমিয়েই মোৰ যেন লগৰী বুকুতে ৰাখো স্মৃতি সোণোৱালী ও গুৱাহাটী সপোন তুমি মোৰ হেপাহ তুমি মোৰ ৱেগ তুমি মোৰ এই জীৱনৰ এই চহৰ আজিও ইমান সজীৱ ইতেও চেনেহে সাৱতিৱ নীৰৱ অধীৰ তোমাতেই শেষ হও অভীষালী ও গুৱাহাটী এই চহৰ আজিও ইমান ৰঙীন ক্য়প্ৰীতিৰ আলোকেৰে ৰয় উজলি তোমাৰেই খ্য়াতি হওক যুগজয়ী ও গুৱাহাটী  সপোন তুমি মোৰ হেপাহ তুমি মোৰ আৱেগ তুমি মোৰ এই জীৱনৰ দিহিঙে দিচাঙে দিবাঙে পাগলাদিয়াৰ পাৰতে দিহিঙে দিচাঙে দিবাঙে পাগলাদিয়াৰ পাৰতে কলং কামেং দৈয়াঙে  একেই মিঠা হাহিৰে আলেঙে আলেঙে আলেঙে পৰ্বতে পাহাৰে ভৈয়ামে আলেঙে আলেঙে আলেঙে পৰ্বতে পাহাৰে ভৈয়ামে হেজাৰ মুখৰ মাজতে একেই মিঠা হাহিৰে দিহিঙে দিচাঙে দিবাঙে পাগলাদিয়াৰ পাৰতে কলং কামেং দৈয়াঙে একেই মিঠা হাহিৰে আলেঙে আলেঙে আলেঙে পৰ্বতে পাহাৰে ভৈয়ামে হেজাৰ মুখৰ মাজতে একেই মিঠা হাহিৰে তুমি কাৰ্বি নে তিৱা তুমি দেউৰী নে ৰাভা তুমি কোচ নে ডিমাচা ও"

hindi_raw_audio = engine.infer_from_text(
    input_text=text,
    lang="as",
    speaker_name="male"
)
byte_io = io.BytesIO()
scipy_wav_write(byte_io, DEFAULT_SAMPLING_RATE, hindi_raw_audio)

with open("assameese_audio_song.wav", "wb") as f:
    f.write(byte_io.read())













