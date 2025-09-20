# Import library
import streamlit as st
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

# ==============================
# 🔑 API Keys
# ==============================
STABILITY_API_KEY = "sk-eRsbgfwpEukz5gPM1W5Qvdrjx0bCb1EOBe2EFt6Fxbfig2QT"  # <-- masukkan API key Stable Diffusion

# --- 1. Page Configuration and Title ---
st.set_page_config(page_title="Krisna Travel Assistant Chatbot", page_icon="🧳", layout="wide")
st.title("🧳 Krisna Travel Assistant Chatbot + Image Generator")
st.caption("Project belajar chatbot AI dari hacktiv 8 mentor by Adipta M")

# --- 2. Sidebar for Settings ---
with st.sidebar:
    st.subheader("⚙️ Settings")
    google_api_key = st.text_input("Google AI API Key", type="password")
    style = st.radio("Gaya Bahasa", ["formal", "santai"])
    reset_button = st.button("🔄 Reset Conversation")

# --- 3. API Key and Agent Initialization ---
if not google_api_key:
    st.info("Masukkan Google AI API key di sidebar untuk mulai chatting.", icon="🗝️")
    st.stop()

if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )
        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[],
            prompt="You are a helpful travel assistant. Always answer clearly and concisely."
        )
        st.session_state._last_key = google_api_key
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.session_state.messages = []
    except Exception as e:
        st.error(f"Invalid API Key or configuration error: {e}")
        st.stop()

# --- 4. Reset Conversation ---
if reset_button:
    st.session_state.pop("agent", None)
    st.session_state.pop("messages", None)
    st.session_state.pop("memory", None)
    st.rerun()

# --- 5. Display Past Messages ---
if "messages" in st.session_state:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ==============================
# 🚀 Fungsi Generate Gambar (Stable Diffusion)
# ==============================
def generate_image_stable_diffusion(prompt: str):
    try:
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/core",
            headers={
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Accept": "application/json"
            },
            files={"none": ''},
            data={
                "prompt": prompt,
                "output_format": "png",
                "aspect_ratio": "1:1"
            },
        )
        if response.status_code != 200:
            return None, f"⚠️ STABILITY_ERROR::{response.text}"

        result = response.json()
        image_base64 = result.get("image")
        if not image_base64:
            return None, "⚠️ Tidak ada gambar dari API"
        return "data:image/png;base64," + image_base64, None
    except Exception as e:
        return None, f"⚠️ STABILITY_ERROR::{e}"

# --- 6. Handle User Input ---
prompt = st.chat_input("Tulis pertanyaanmu tentang perjalanan...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # Jika user minta gambar
        if any(word in prompt.lower() for word in ["gambar", "foto", "lukisan", "image", "illustration"]):
            with st.chat_message("assistant"):
                st.markdown("🎨 Sedang membuat gambar...")
            image_url, error = generate_image_stable_diffusion(prompt)
            if image_url:
                with st.chat_message("assistant"):
                    st.image(image_url, caption=f"🖼️ {prompt}", use_container_width=True)
                st.session_state.messages.append({"role": "assistant", "content": f"(Gambar dibuat dari prompt: {prompt})"})
            else:
                with st.chat_message("assistant"):
                    st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
        else:
            # Tambahkan gaya bahasa ke prompt
            style_instruction = (
                "Jawablah dengan gaya bahasa formal, sopan, terstruktur."
                if style == "formal"
                else "Jawablah dengan gaya bahasa santai, ringan, seperti ngobrol dengan teman."
            )
            full_prompt = f"{prompt}\n\n{style_instruction}"

            # Memory input
            memory = st.session_state.memory
            messages = memory.chat_memory.messages + [HumanMessage(content=full_prompt)]

            response = st.session_state.agent.invoke({"messages": messages})

            if response and "messages" in response and len(response["messages"]) > 0:
                answer = response["messages"][-1].content
            else:
                answer = "Maaf, saya tidak bisa menjawab saat ini."

            # Simpan ke memory & chat history
            memory.chat_memory.add_user_message(prompt)
            memory.chat_memory.add_ai_message(answer)

            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        answer = f"⚠️ Terjadi error: {e}"
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
