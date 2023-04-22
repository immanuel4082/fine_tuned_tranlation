import streamlit as st
from mtranslate import translate
import pandas as pd
import os
from gtts import gTTS
import base64
import pandas as pd
from transformers import pipeline,AutoTokenizer, AutoModelForSeq2SeqLM


# Load a pretrained tokenizer for the source and target languages
tokenizer = AutoTokenizer.from_pretrained("KigenCHESS/fine_tuned_eng-sw")

# load the model 
model = AutoModelForSeq2SeqLM.from_pretrained("KigenCHESS/fine_tuned_eng-sw", from_tf=True)

# Set up the translation pipeline using the loaded model
translator = pipeline("translation", model=model, tokenizer=tokenizer)

# layout
st.title("Language-Translation")
st.markdown("In Python 🐍 with Streamlit")
st.markdown("by DR Andrew Kipkebut")
inputtext = st.text_area("INPUT",height=200)

#the correct translation 
speech_lang = {
    "sw": "Swahili",
}

selected_lang = None
for lang_code, lang_name in speech_lang.items():
    if st.button(lang_name):
        selected_lang = lang_code
        break

#to create two columns 
c1,c2 = st.columns([4,3])

#I/0
if len(inputtext) > 0 :
    try:
        output = translator(inputtext)
        translated_text = output[0]['translation_text']
        with c1:
            st.text_area("PREDICTED TRANSLATED TEXT", translated_text, height=200)

        #the translation below is the correct 
        output = translate(inputtext,selected_lang)
        with c2:
            st.text_area("CORRECT TRANSLATED TEXT",output,height=200)
    except Exception as e:
        st.error(e)
