import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from load_model import load_model

st.title("BASIC RAG INSURANCE")


def generate_response(input_text):
    model = load_model(
        "/Users/user/Documents/personal_projects/rag_insurance/model/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
    )
    st.info(model.invoke(input_text))

    # Allow the user to upload a PDF


uploaded_file = st.file_uploader(label="Upload your PDF")

if uploaded_file:
    working_dir = "/Users/user/Documents/personal_projects/rag_insurance"
    file_path = f"{working_dir}/{uploaded_file.name}"
    # Save the uploaded file to the working directory
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    pdf_viewer(file_path)


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
