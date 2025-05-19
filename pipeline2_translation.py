import torch
from IndicTransToolkit.processor import IndicProcessor 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



print("Loading model...")

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

src_lang, tgt_lang = "hin_Deva", "tam_Taml"
model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    attn_implementation="flash_attention_2"
).to(DEVICE)

ip = IndicProcessor(inference=True)



input_sentences = [
    'जब मैं छोटा था, मैं हर दिन पार्क जाता था।',
    'हमने पिछले हफ्ते एक नई फिल्म देखी, जो बहुत प्रेरणादायक थी।',
    'अगर आप उस समय मुझसे मिले होते, तो हम खाने के लिए बाहर गए होते।',
    'मेरे दोस्त ने मुझे अपने जन्मदिन की पार्टी में आमंत्रित किया है, और मैं उसे एक उपहार दूंगा।'
]

batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True,
).to(DEVICE)


with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        use_cache=True,
        min_length=0,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
    )


generated_tokens = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)


translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

for input_sentence, translation in zip(input_sentences, translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")
