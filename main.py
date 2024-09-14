import streamlit as st
from audio_recorder_streamlit import audio_recorder
from io import BytesIO
from groq import Groq
from decouple import config
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from elevenlabs import VoiceSettings, play
from elevenlabs.client import ElevenLabs

# Load API keys
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = config("GROQ_API_KEY")
    st.session_state.eleven_api_key = config("ELEVEN_API_KEY")

groq_api_key = st.session_state.groq_api_key
eleven_api_key = st.session_state.eleven_api_key
groq_client = Groq(api_key=groq_api_key)
elevenlabs_client = ElevenLabs(api_key=eleven_api_key)

# Initialize Groq model
model = ChatGroq(model="llama3-8b-8192")

# Config for session management
config = {"configurable": {"session_id": "abc2"}}

def main():
    # Set page config
    st.set_page_config(page_title='Groq voicebot', page_icon='🎤')

    # Custom styling
    st.markdown("""
    <style>
    .user-message { background-color: #3E4C59; color: white; padding: 10px; border-radius: 10px; margin: 5px; }
    .assistant-message { background-color: #FFD700; color: black; padding: 10px; border-radius: 10px; margin: 5px; }
    .stButton>button { background-color: #007bff; color: pink; font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)

    st.title('Groq voiceBot')

    system_prompt = 'You are a friendly assistant. Keep your responses short.'
    
    # Initialize greeting and chat messages
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
        st.write("Hello! How can I assist you today?")

    # Initialize ChatMessageHistory in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

    # Retrieve chat history
    chat_history = st.session_state.chat_history

    # st.write(chat_history)
    audio_bytes = audio_recorder(text="")

    # Display chat history
    st.markdown("### Conversation:")
    for message in st.session_state.chat_messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'><strong>User:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)

    # Handle audio input
    if audio_bytes:
        audio_file_like = BytesIO(audio_bytes)
        user_question = speech_to_text(audio_file_like)

        if user_question:
            # Store user question in session state and chat history
            st.session_state.chat_messages.append({"role": "user", "content": user_question})

            # Display user question
            st.markdown(f"<div class='user-message'><strong>User:</strong> {user_question}</div>", unsafe_allow_html=True)

            # Create the chat prompt
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history")  # Incorporates chat history
            ])

            # Combine the prompt with the model and history
            chain = prompt | model
            with_message_history = RunnableWithMessageHistory(chain, lambda: chat_history)

            # Get assistant's response
            response = with_message_history.invoke([HumanMessage(content=user_question)], config=config)
            assistant_response = response.content

            

            # Store assistant response in session state and chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})

            # Display assistant response
            st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong> {assistant_response}</div>", unsafe_allow_html=True)

            text_to_speech_stream(assistant_response)
            

def speech_to_text(audio_bytes_io):
    transcription = groq_client.audio.transcriptions.create(
        file=("audio.wav", audio_bytes_io.read()),
        model="distil-whisper-large-v3-en",
        prompt="You are converting speech to text",
        response_format="json",
        language="en"
    )
    return transcription.text

def text_to_speech_stream(text: str):
    response = elevenlabs_client.text_to_speech.convert(
        voice_id="21m00Tcm4TlvDq8ikWAM",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(stability=0.7, similarity_boost=1.0, style=0.0, use_speaker_boost=True)
    )
    audio_stream = BytesIO()
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)
    play(audio_stream.getvalue())

if __name__ == "__main__":
    main()
