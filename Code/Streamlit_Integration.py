import streamlit as st
import pymysql
import boto3
import pandas as pd
from io import BytesIO
from PIL import Image
import zipfile
from PyPDF2 import PdfReader
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

# Load pre-trained model for sentence embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

openai.api_key = "sk-rcRLfezEqYqT76yYsFtT9E_QUWGBgheWXZOMVHpUNvT3BlbkFJh92hu_PqcVOaUo75GwdCrT4TJOUumpb65DYqgMRzgA"

# Mysql connection
def connect_mysql():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='admin0077',
        database='GAIA',
    )
    return connection

# querying from Mysql
def get_questions():
    connection = connect_mysql()
    cursor = connection.cursor()
    query = "SELECT task_id, question FROM METADATA"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result

# S3 connection
def get_files_from_s3(task_id):
    s3 = boto3.client('s3', 
                      aws_access_key_id='AKIAZQ3DQKBOTUPXE3TZ', 
                      aws_secret_access_key='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', 
                      region_name='us-east-2')
    bucket_name = 'gaia-benchmark-dataset'
    prefix = f'GAIA/2023/validation\\{task_id}'

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if 'Contents' in response:
        file_names = [content['Key'] for content in response['Contents']]
        return file_names
    else:
        return []

# Display contents in S3
def display_file_from_s3(file_key):
    s3 = boto3.client('s3', 
                      aws_access_key_id='AKIAZQ3DQKBOTUPXE3TZ', 
                      aws_secret_access_key='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', 
                      region_name='us-east-2')
    bucket_name = 'gaia-benchmark-dataset'

    # Fetching objects from S3
    file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    file_content = file_obj['Body'].read()

    # Displaying file based on format
    if file_key.endswith('.mp3'):
        st.audio(file_content, format='audio/mp3')
    elif file_key.endswith(('.jpg', '.jpeg', '.png')):
        image = Image.open(BytesIO(file_content))
        st.image(image, caption=file_key)
    elif file_key.endswith('.pdf'):
        # Display PDF preview
        reader = PdfReader(BytesIO(file_content))
        num_pages = len(reader.pages)
        st.write(f"**PDF Preview - {file_key}:**")
        for page_num in range(min(3, num_pages)):  # first 3 pages 
            page = reader.pages[page_num]
            st.write(page.extract_text())
    elif file_key.endswith('.xlsx'):
        # Display Excel preview using pandas
        excel_data = pd.read_excel(BytesIO(file_content), sheet_name=None) 
        for sheet_name, df in excel_data.items():
            st.write(f"**Excel Preview - {sheet_name} (Sheet):**")
            st.dataframe(df.head())  
    elif file_key.endswith('.zip'):
        # Preview ZIP contents 
        with zipfile.ZipFile(BytesIO(file_content), 'r') as z:
            st.write(f"**ZIP Archive - {file_key}:**")
            zip_files = z.namelist()
            st.write(zip_files)  
    else:
        st.write("Unsupported file type:", file_key)

def get_database_answer(task_id):
    connection = connect_mysql()
    cursor = connection.cursor()
    query = "SELECT final_answer FROM METADATA WHERE task_id = %s"
    cursor.execute(query, (task_id,))
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    return result

def get_annotator_steps(task_id):
    connection = connect_mysql()
    cursor = connection.cursor()
    query = "SELECT annotator_steps FROM METADATA WHERE task_id = %s"
    cursor.execute(query, (task_id,))
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    return result

def clean_text(text):
    # Remove extra spaces, special characters, normalize punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()

def cosine_similarity_embeddings(openai_answer, database_answer):
    # Preprocess both answers
    openai_cleaned = clean_text(openai_answer)
    database_cleaned = clean_text(database_answer)
    
    # Generate sentence embeddings
    openai_embedding = model.encode([openai_cleaned])
    database_embedding = model.encode([database_cleaned])
    
    # Compute cosine similarity
    similarity = cosine_similarity(openai_embedding, database_embedding)
    similarity_score = similarity[0][0]
    
    return similarity_score

def compare_answers(openai_answer, database_answer, threshold=0.8):
    # Step 1: Check if database answer is a single word and if it appears in the generated answer
    database_answer_cleaned = clean_text(database_answer)
    
    if len(database_answer_cleaned.split()) == 1:  # If the database answer is just one word
        if database_answer_cleaned in clean_text(openai_answer):
            # If the single-word database answer exists in the OpenAI answer, return True
            st.write("Exact match found for single-word database answer.")
            return True
    
    # Step 2: Fall back to cosine similarity if no direct match was found
    similarity = cosine_similarity_embeddings(openai_answer, database_answer)
    return similarity >= threshold

# Streamlit main page
def main():
    st.set_page_config(page_title="Task Viewer and OpenAI Answer", layout="wide")

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'main'
    if st.session_state.page == 'main':
        main_page()
    elif st.session_state.page == 'compare_page':
        compare_page()
    elif st.session_state.page == 'regenerate_page':
        regenerate_answer_page()

def main_page():
    # Custom CSS for black background and white text
    st.markdown("""
        <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stSelectbox label {
            font-size: 2.2em !important;
            color: white !important;
        }
        .stTextInput>div>div>input, .stSelectbox>div>div>div>input {
            color: black;
        }
        h1 {
            text-align: center;
            font-size: 3.5em;
            color: #FFD700;
            padding-top: 1em;
            padding-bottom: 0.5em;
        }
        .stSelectbox, .stDataFrame {
            background-color: #222222;
            border-radius: 10px;
        }
        .stMarkdown {
            color: white;
        }
        .stButton button {
            background-color: #FFD700;
            color: black;
        }
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
        }
        .main, .block-container {
            background-color: #121212;  /* This ensures the whole background is black */
        }
        .stMarkdown h2 {
            color: #FFD700;
        }
        
        label {
            font-size: 1.25em; /* Adjust the font size (4 points larger) */
            color: white;      /* White color for the label */
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Task Viewer")

    # Fetch questions from MySQL
    question_data = get_questions()

    # Numbering the questions
    questions = [f"{i+1}. {item[1]}" for i, item in enumerate(question_data)]
    question_map = {f"{i+1}. {item[1]}": item[0] for i, item in enumerate(question_data)}

    # Dropdown to select a question
    selected_question = st.selectbox("Select a question:", questions)

    if selected_question:
        # Get corresponding task_id
        selected_task_id = question_map[selected_question]

        # Fetch file names from S3 based on task_id
        file_names = get_files_from_s3(selected_task_id)

        # Display the question and the associated files from S3
        st.write(f"**Selected Question:** {selected_question}")
        st.write(f"**Task ID:** {selected_task_id}")
        
        if file_names:
            st.write(f"**Files in S3 for Task ID {selected_task_id}:**")
            for file_name in file_names:
                st.write(f"- {file_name}")
                display_file_from_s3(file_name)
        else:
            st.markdown(
                """
                <div style="text-align: center; animation: blinker 1.5s linear infinite; color: red; font-weight: bold;">
                    ⚠️ No additional context found for this question ⚠️
                </div>
                <style>
                @keyframes blinker {
                    50% { opacity: 0; }
                }
                </style>
                """, 
                unsafe_allow_html=True
            )

        st.session_state.selected_question = selected_question
        st.session_state.associated_files = file_names
        st.session_state.selected_task_id = selected_task_id

    # Generate Answer and Compare on Main Page
    if st.button("Generate Answer"):
        try:
            prompt = f"Question: {st.session_state.selected_question}\n"
            if 'associated_files' in st.session_state and st.session_state.associated_files:
                prompt += "Associated files:\n"
                for file in st.session_state.associated_files:
                    prompt += f"- {file}\n"
            prompt += "\nPlease provide an answer to the question based on the available information."

            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=150
            )
            openai_answer = response.choices[0].text.strip()
            st.session_state.openai_answer = openai_answer

            # Fetch database answer and compare
            database_result = get_database_answer(st.session_state.selected_task_id)
            if database_result:
                database_answer = database_result[0]
                st.session_state.database_answer = database_answer

                is_correct = compare_answers(openai_answer, database_answer)
                st.session_state.is_correct = is_correct

            st.session_state.page = 'compare_page'
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {str(e)}")

def compare_page():
    st.title("Open AI Answer and Comparison")
    
    if 'selected_question' in st.session_state:
        st.write(f"**Selected Question:** {st.session_state.selected_question}")
        st.write(f"**OpenAI Answer:** {st.session_state.openai_answer}")
        st.write(f"**Database Answer:** {st.session_state.database_answer}")

        if st.session_state.is_correct:
            st.success("The OpenAI answer matches the database answer.")
        else:
            st.error("The OpenAI answer does not match the database answer.")

    # Navigation buttons
    if st.button("Go Back"):
        st.session_state.page = 'main'
        st.rerun()

    if st.button("Regenerate Answer"):
        st.session_state.page = 'regenerate_page'
        st.rerun()

def regenerate_answer_page():
    st.title("Regenerated Answer")

    if 'selected_question' in st.session_state:
        st.write(f"**Selected Question:** {st.session_state.selected_question}")

        # Prepare the regeneration prompt with annotator steps and files
        annotator_steps_result = get_annotator_steps(st.session_state.selected_task_id)
        if annotator_steps_result:
            annotator_steps = annotator_steps_result[0]
            prompt = f"Question: {st.session_state.selected_question}\n"
            prompt += f"Annotator Steps: {annotator_steps}\n"
            if 'associated_files' in st.session_state and st.session_state.associated_files:
                prompt += "Associated files:\n"
                for file in st.session_state.associated_files:
                    prompt += f"- {file}\n"
            prompt += "\nPlease regenerate the answer based on the provided steps and files."

            # Call OpenAI API for regenerated answer
            try:
                response = openai.Completion.create(
                    engine="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=200
                ) 
                regenerated_answer = response.choices[0].text.strip()
                st.write("**Regenerated OpenAI Answer:**")
                st.write(regenerated_answer)

                # Display annotator steps and object name from S3
                st.write(f"**Annotator Steps:** {annotator_steps}")
                if 'associated_files' in st.session_state:
                    st.write(f"**Object from S3 used:** {st.session_state.associated_files}")

                # Compare regenerated answer with the database answer
                if 'database_answer' in st.session_state:
                    is_correct = compare_answers(regenerated_answer, st.session_state.database_answer)
                    if is_correct:
                        st.success("The regenerated answer matches the database answer.")
                    else:
                        st.error("The regenerated answer does not match the database answer.")
            except Exception as e:
                st.error(f"An error occurred while regenerating the answer: {str(e)}")
        else:
            st.warning("No annotator steps found for this question.")

    # Add navigation buttons
    if st.button("Go Back"):
        st.session_state.page = 'compare_page'
        st.rerun()

    if st.button("Go Home"):
        st.session_state.page = 'main'
        st.rerun()

if __name__ == "__main__":
    main()
