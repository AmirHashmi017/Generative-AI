from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
prompt_template=load_prompt("prompt_template.json")
model= ChatOpenAI(model='gpt-4')

# Web app UI
st.header("Research Tool")

paper_input= st.selectbox("Select the Research Paper Name",
                          ["Attention Is All You Need",
                            "BERT: Pre-training of Deep Bidirectional Transformers",
                            "GPT-3: Langauage Models are Few-SHot Learners",
                            "Diffusion Models Beat GANs on Image Synthesis"])

style_input=st.selectbox("Select Explanation Style",["Beginner-Friendly","Technical","Core-Oriented",
                                                     "Mathematical"])

length_input= st.selectbox("Select Explanation Length",["Short (1-2 paragraphs)",
                            "Medium (3-5 paragraphs)","Long (detailed explanation)"])



if st.button("Summarize"):
    prompt= prompt_template.invoke({
    "paper_input":paper_input,
    "style_input": style_input,
    "length_input":length_input
    })
    result= model.invoke(prompt)
    st.write(result.content)

