import streamlit as st
from audio_recorder_streamlit import audio_recorder
from io import BytesIO
from groq import Groq
from decouple import config
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from elevenlabs import VoiceSettings,play
from elevenlabs.client import ElevenLabs


if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = config("GROQ_API_KEY")
    st.session_state.eleven_api_key = config("ELEVEN_API_KEY")

# Api key creation
groq_api_key = st.session_state.groq_api_key
eleven_api_key = st.session_state.eleven_api_key
groq_client = Groq(api_key=groq_api_key)
elevenlabs_client = ElevenLabs(api_key=eleven_api_key,)

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
    
    st.title('Groq voiceBot')

    system_prompt = 'You are a friendly assistant.Keep your responses short'
    greeting="Hi! How can i assist you today?"
    
    conversational_memory_length = 5

    # Initialize the Groq model
    if 'groq_chat' not in st.session_state:
        st.session_state.groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name='llama3-8b-8192'
    )

    groq_chat = st.session_state.groq_chat

    # Initialize session state for conversation history
    if 'chat_messages' not in st.session_state:
        st.write(greeting)
        st.session_state.chat_messages = []  # Store individual chat messages

    # Store memory in session state to persist across multiple runs
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    memory = st.session_state.memory

    st.markdown("### Click to ask question:")
    audio_bytes = audio_recorder(text="",)

    # Display chat history
    st.markdown("### Conversation:")
    for message in st.session_state.chat_messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'><strong>User:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)
    
    if audio_bytes:
        audio_file_like = BytesIO(audio_bytes)
        user_question = speech_to_text(audio_file_like)

        if user_question:

            # Append user question
            st.session_state.chat_messages.append({"role": "user", "content": user_question})

            # Display user question
            st.markdown(f"<div class='user-message'><strong>User:</strong> {user_question}</div>", unsafe_allow_html=True)

            # Construct a chat prompt template using various components
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),  # Retains the chat history
                    HumanMessagePromptTemplate.from_template("{human_input}")
                ]
            )

            # Create a conversation chain using LangChain LLM
            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=False,
                memory=memory  # Use the persistent memory
            )

            # Get the chatbot's response
            response = conversation.predict(human_input=user_question)
            
            # Append assistant response
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

            # Display assistant response
            st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong> {response}</div>", unsafe_allow_html=True)

            # Generate audio response
            text_to_speech_stream(response)

            
        
def speech_to_text(audio_bytes_io): 
    transcription = groq_client.audio.transcriptions.create(
        file=("audio.wav", audio_bytes_io.read()),
        model="distil-whisper-large-v3-en", # WhisperModel
        prompt="You are converting speech to text",
        response_format="json",
        language="en",
        )
    return transcription.text


def text_to_speech_stream(text: str):
    response = elevenlabs_client.text_to_speech.convert(
        voice_id="21m00Tcm4TlvDq8ikWAM", 
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2", # Eleven model
        voice_settings=VoiceSettings(
            stability=0.7,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Create a BytesIO object to hold the audio data in memory
    audio_stream = BytesIO()

    # Write each chunk of audio data to the stream
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

     # Convert the BytesIO stream to bytes
    audio_data = audio_stream.getvalue()
    play(audio_data)

    

if __name__ == "__main__":
    main()