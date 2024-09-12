import streamlit as st
from audio_recorder_streamlit import audio_recorder

def main():
    # Set page config
    st.set_page_config(page_title='Groq voicebot', page_icon='🎤')
    # Custom styling
    st.markdown("""
    <style>
    .user-message {
        background-color: #3E4C59;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
    .assistant-message {
        background-color: #FFD700;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
    .stButton>button {
        background-color: #007bff;
        color: pink;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title('Groq VoiceBot')

    audio_bytes = audio_recorder(text="",)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with open('audio.wav', mode='wb') as f:
            f.write(audio_bytes)

if __name__ == "__main__":
    main()