import streamlit as st
from google import genai
from google.genai import types
from datetime import datetime


# Title
st.title("🤖 Gemini AI Chatbot")
st.markdown("Chat with Google's Gemini AI model!")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # API key input
    api_key = st.text_input(
        "Enter your Gemini API Key:",
        type="password",
        help="Get your API key from https://aistudio.google.com/app/apikey"
    )
    
    # Model selection
    model_name = st.selectbox(
        "Select Model:",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(f"*{message['timestamp']}*")

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar to start chatting.")
        st.stop()
    
    # Add user message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"*{timestamp}*")
    
    # Generate AI response
    with st.chat_message("assistant"):
        try:
            # Prepare conversation history
            contents = []
            for msg in st.session_state.messages:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })

            # Initialize client
            client = genai.Client(api_key=api_key)
            
            # Stream the response
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    temperature=1,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            
            # Display streaming response
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
            
            # Final response without cursor
            response_placeholder.markdown(full_response)
            
            # Add timestamp
            response_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"*{response_timestamp}*")
            
            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "timestamp": response_timestamp
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your API key and try again.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p><small>Get your free API key at "
    "<a href='https://aistudio.google.com/app/apikey' target='_blank'>Google AI Studio</a></small></p>"
    "</div>", 
    unsafe_allow_html=True
)
