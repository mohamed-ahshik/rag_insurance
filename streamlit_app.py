import json

import requests
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from preprocess import run_preprocess

st.title("Retrieval Augmented Generation")


def generate_response(input_text: str) -> None:
    """
    Generate a response using the DeepSeek API

    Parameters:
    input_text (str): The input text

    Returns:
    None
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "deepseek-r1:7b",
        "prompt": input_text,
        "stream": False,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_dict = response.json()
    st.info(response_dict["response"])

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
        "Type your question here",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        compress_retriever = run_preprocess(file_path)
        compress_docs = compress_retriever.invoke(text)
        print(compress_docs)
        prompt = f"""You are an assistant
        Understanding the question and providing the answer is your job.
        Answer the following question based on the information provided in the documents.
        Be clear and verbose in your response.
        Here is the question:
        {text}
        Only reply from the following documents:
        <document>
        {compress_docs}
        </document>
        If you don't know the answer, please say "I don't know"
        ## OUTPUT :
        - Output in markdown format only.
        """
        generate_response(prompt)
