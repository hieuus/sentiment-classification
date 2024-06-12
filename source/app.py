import streamlit as st
from streamlit_extras.let_it_rain import rain
# import pytorch
import random as random
from transformers import pipeline
from preprocessing import cleanData

button_style = '''
    <style>
        .stButton button {
            
            background-color: #0072B1;
            color: white;
            border-radius: 5px;
            font-weight: bold;
            padding: 8px 16px;
            box-shadow: none;
            
        }
        .stButton button:hover {
            color: white;
            background-color: #0072B1;
            box-shadow: none;
            border: none;
        }
    </style>'''


st.set_page_config(page_title="Sentiment Classification")

st.image('source/assets/sentiment.png')
st.title("Sentiment Classification")
st.markdown(button_style, unsafe_allow_html=True)


# Sidebar
st.sidebar.title("Team Information")
st.sidebar.write("Members:")
st.sidebar.write("- 20120084 - Nguyễn Văn Hiếu")
st.sidebar.write("- 20120085 - Trần Xuân Hòa")

# Tiêu đề nhập văn bản
st.header("Input Tweet")


# Ô input để nhập văn bản
context = st.text_area("Please type the tweet in the blank field")
context_processing=cleanData(context)

def toSentiment(sentinent):
    if sentinent == 'LABEL_1':
        return 'Neutral'
    elif sentinent == 'LABEL_2':
        return 'Positive'
    else:
        return 'Negative'
    

def getIcon(status):
    if status == "Negative":
        return "😡"
    if status == "Positive":
        return "😍"
    else:
        return "😐"


# Ô kết quả trả về đúng/sai
if st.button("Predict"):
    model_ckpt='model'
    pipe=pipeline('sentiment-analysis',model=model_ckpt)

    result = toSentiment(pipe(context_processing)[0]['label'])
    score = pipe(context_processing)[0]['score']
    
    st.write('Result: ' +  result)
    st.write('Score: ' + str(score))

    icon = getIcon(result)
    rain(emoji=icon, font_size=64, falling_speed=5, animation_length=1)



    