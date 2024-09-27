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
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-trained model for sentence embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

openai.api_key = "sk-rcRLfezEqYqT76yYsFtT9E_QUWGBgheWXZOMVHpUNvT3BlbkFJh92hu_PqcVOaUo75GwdCrT4TJOUumpb65DYqgMRzgA"

# Mysql connection
def connect_mysql():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='Siddhivinayak@8',
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

def reset_attempts_and_validation():
    connection = connect_mysql()
    cursor = connection.cursor()
    query = "UPDATE METADATA SET Number_of_Attempts = 0, Validated = 'No'"
    cursor.execute(query)
    connection.commit()
    cursor.close()
    connection.close()

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

def update_attempts(task_id):
    connection = connect_mysql()
    cursor = connection.cursor()
    query = "UPDATE METADATA SET Number_of_Attempts = Number_of_Attempts + 1 WHERE task_id = %s"
    cursor.execute(query, (task_id,))
    connection.commit()
    cursor.close()
    connection.close()

def update_validation(task_id, validated):
    connection = connect_mysql()
    cursor = connection.cursor()
    query = "UPDATE METADATA SET Validated = %s WHERE task_id = %s"
    cursor.execute(query, (validated, task_id))
    connection.commit()
    cursor.close()
    connection.close()

def get_visualization_data():
    connection = connect_mysql()
    cursor = connection.cursor()
    query = "SELECT task_id, task_no, Number_of_Attempts, Validated FROM METADATA ORDER BY task_no"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    
    df = pd.DataFrame(result, columns=['task_id', 'Task_No', 'Number_of_Attempts', 'Validated'])
    return df

def compare_answers(openai_answer, database_answer, task_id, threshold=0.8):
    # Clean both answers
    openai_cleaned = clean_text(openai_answer)
    database_cleaned = clean_text(database_answer)

    # Check if database answer (whether single-word or phrase) is a substring of OpenAI answer
    if database_cleaned in openai_cleaned:
        update_validation(task_id, 'Yes')
        st.write("Exact match found in OpenAI answer.")
        return True

    # Fall back to cosine similarity if no direct match was found
    similarity = cosine_similarity_embeddings(openai_cleaned, database_cleaned)
    if similarity >= threshold:
        update_validation(task_id, 'Yes')
        return True
    
    # If no match
    update_validation(task_id, 'No')
    return False

def attempts_count_chart(data):
    attempts_count = data['Number_of_Attempts'].value_counts().sort_index()
    plt.figure(figsize=(12, 8))
    bars = plt.bar(attempts_count.index, attempts_count.values, color='#1f77b4')
    plt.title('Number of Attempts vs Count of Task IDs', fontsize=16)
    plt.xlabel('Number of Attempts', fontsize=12)
    plt.ylabel('Count of Task IDs', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout()
    st.pyplot(plt)

def correct_responses_chart(data):
    first_attempt = data[(data['Number_of_Attempts'] == 1) & (data['Validated'] == 'Yes')].shape[0]
    regenerated_attempts = data[(data['Number_of_Attempts'] > 1) & (data['Validated'] == 'Yes')].shape[0]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['First Attempt', 'Regenerated Attempts'], [first_attempt, regenerated_attempts], color=['#3C99DC', '#0F5298'])
    plt.title('Correct Responses: First Attempt vs Regenerated Attempts', fontsize=16)
    plt.ylabel('Count', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height}', 
                 ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    st.pyplot(plt)

def validation_status_chart(data):
    validated_count = data[data['Validated'] == 'Yes'].shape[0]
    not_validated_count = data[data['Validated'] == 'No'].shape[0]
    
    # Handle potential NaN values
    total_count = validated_count + not_validated_count
    if total_count == 0:
        st.write("No data available for validation status chart.")
        return

    plt.figure(figsize=(10, 10))
    colors = ['#3C99DC', '#0F5298']
    plt.pie([validated_count, not_validated_count], 
            labels=['Validated', 'Not Validated'], 
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*total_count):d})',
            colors=colors, 
            textprops={'fontsize': 12})
    plt.title('Validation Status of Task IDs', fontsize=16)
    plt.tight_layout()
    st.pyplot(plt)
        
# Streamlit main page
def main():
    st.set_page_config(page_title="Task Genie: AI-Powered Answers", layout="wide")
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Amasis+MT+Pro&display=swap');
        html, body, [class*="css"] {
            font-family: 'Amasis MT Pro', serif;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Reset the Number_of_Attempts and validation on app close or refresh
    if 'first_run' not in st.session_state:
        reset_attempts_and_validation()  
        st.session_state.first_run = True  # Only reset once during the app session

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'main'
    if st.session_state.page == 'main':
        main_page()
    elif st.session_state.page == 'compare_page':
        compare_page()
    elif st.session_state.page == 'regenerate_page':
        regenerate_answer_page()
    elif st.session_state.page == 'visualizations':
        visualizations_page()

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
            font-family: "Times New Roman", Arial, sans-serif;
            text-align: center;
            font-size: 2.75em;
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

    st.title("Task Genie : AI Powered Answers")

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
        st.markdown(f"<u>**Selected Question:**</u> {selected_question}", unsafe_allow_html=True)
        st.markdown(f"<u>**Task ID:**</u> {selected_task_id}", unsafe_allow_html=True)
        
        if file_names:
            st.markdown(f"<u>**Files in S3 for Task ID**</u> {selected_task_id}:", unsafe_allow_html=True)
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
    if st.button("Generate Response"):
        try:
            update_attempts(st.session_state.selected_task_id)
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

                is_correct = compare_answers(openai_answer, database_answer, st.session_state.selected_task_id)
                st.session_state.is_correct = is_correct

            st.session_state.page = 'compare_page'
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {str(e)}")

    # Add the Visualizations button
    if st.button("Visualizations"):
        st.session_state.page = 'visualizations'
        st.rerun()

def compare_page():
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
            font-family: "Times New Roman", Arial, sans-serif;
            text-align: center;
            font-size: 2.75em;
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

    st.title("Open AI Response Comparison")
    
    if 'selected_question' in st.session_state:
        st.markdown(f"<u>**Selected Question:**</u> {st.session_state.selected_question}", unsafe_allow_html=True)
        st.markdown(f"<u>**OpenAI Answer:**</u> {st.session_state.openai_answer}", unsafe_allow_html=True)
        st.markdown(f"<u>**Database Answer:**</u> {st.session_state.database_answer}", unsafe_allow_html=True)
       
        if st.session_state.is_correct:
            st.markdown(
                """
                <div style="text-align: center; animation: blinker 1.5s linear infinite; color: Green; font-weight: bold;">
                    Open AI response matches the database answer!
                </div>
                <style>
                @keyframes blinker {
                    50% { opacity: 0; }
                }
                </style>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="text-align: center; animation: blinker 1.5s linear infinite; color: red; font-weight: bold;">
                    ⚠️ Open AI response doesn`t match the database answer ⚠️
                </div>
                <style>
                @keyframes blinker {
                    50% { opacity: 0; }
                }
                </style>
                """, 
                unsafe_allow_html=True
            )

    if st.button("Regenerate Response"):
        update_attempts(st.session_state.selected_task_id)  # Increment count here
        st.session_state.page = 'regenerate_page'
        st.rerun()
    
    # Navigation buttons
    if st.button("Go Back"):
        st.session_state.page = 'main'
        st.rerun()

def regenerate_answer_page():
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
            font-family: "Times New Roman", Arial, sans-serif;
            text-align: center;
            font-size: 2.75em;
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

    st.title("Regenerated Response")

    # Ensure the question is available in session state
    if 'selected_question' in st.session_state:
        st.markdown(f"<u>**Selected Question:**</u> {st.session_state.selected_question}", unsafe_allow_html=True)

        # Prepare the prompt for OpenAI using the question, annotator steps, and associated files
        annotator_steps_result = get_annotator_steps(st.session_state.selected_task_id)
        if annotator_steps_result:
            annotator_steps = annotator_steps_result[0]
            prompt = f"Question: {st.session_state.selected_question}\n"
            prompt += f"Annotator Steps: {annotator_steps}\n"
            if 'associated_files' in st.session_state and st.session_state.associated_files:
                prompt += "Associated files:\n"
                for file in st.session_state.associated_files:
                    prompt += f"- {file}\n"
            prompt += "\nPlease generate an answer based on the provided steps and files."

            # Automatically generate the OpenAI answer and display it directly
            try:
                response = openai.Completion.create(
                    engine="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=200
                )
                openai_answer = response.choices[0].text.strip()
                st.session_state.openai_answer = openai_answer
                st.markdown(f"<u>**Task ID:**</u> {st.session_state.selected_task_id}", unsafe_allow_html=True)

                # Display task details, annotator steps, and files
                st.markdown(f"<u>**Annotator Steps:**</u> {annotator_steps}", unsafe_allow_html=True)
                if 'associated_files' in st.session_state:
                    st.markdown(f"<u>**Associated Files:**</u> {st.session_state.associated_files}", unsafe_allow_html=True)
                st.markdown(f"<u>**OpenAI Generated Answer:**</u> {openai_answer}", unsafe_allow_html=True)
                
                # Compare the OpenAI answer with the database answer
                if 'database_answer' in st.session_state:
                    is_correct = compare_answers(openai_answer, st.session_state.database_answer, st.session_state.selected_task_id)
                    st.session_state.is_correct = is_correct

                    if is_correct:
                        st.markdown(
                        """
                        <div style="text-align: center; animation: blinker 1.5s linear infinite; color: Green; font-weight: bold;">
                            Open AI response matches the database answer!
                        </div>
                        <style>
                        @keyframes blinker {
                            50% { opacity: 0; }
                        }
                        </style>
                        """, 
                        unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                        """
                        <div style="text-align: center; animation: blinker 1.5s linear infinite; color: red; font-weight: bold;">
                            ⚠️ Open AI response doesn`t match the database answer ⚠️
                        </div>
                        <style>
                        @keyframes blinker {
                            50% { opacity: 0; }
                        }
                        </style>
                        """, 
                        unsafe_allow_html=True
                        )
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {str(e)}")

        else:
            st.warning("No annotator steps found for this question.")

        # Display buttons based on whether the answer matches the database answer
        if 'is_correct' in st.session_state and st.session_state.is_correct:
            if st.button("Go Home", key="go_home_button"):
                st.session_state.page = 'main'
                st.rerun()
        else:
            # The regenerate answer button will keep showing until the correct match is found
            if st.button("Regenerate Answer", key="regenerate_button"):
                update_attempts(st.session_state.selected_task_id)  # Increment count here as well
                
                try:
                    # Regenerate the answer
                    response = openai.Completion.create(
                        engine="gpt-3.5-turbo-instruct",
                        prompt=prompt,
                        max_tokens=200
                    )
                    regenerated_answer = response.choices[0].text.strip()
                    st.session_state.openai_answer = regenerated_answer
                    st.write("**Regenerated OpenAI Answer:**")
                    st.write(regenerated_answer)

                    # Compare regenerated answer with database answer
                    if 'database_answer' in st.session_state:
                        is_correct = compare_answers(regenerated_answer, st.session_state.database_answer, st.session_state.selected_task_id)
                        st.session_state.is_correct = is_correct

                        if is_correct:
                            st.markdown(
                            """
                            <div style="text-align: center; animation: blinker 1.5s linear infinite; color: Green; font-weight: bold;">
                                Open AI response matches the database answer!
                            </div>
                            <style>
                            @keyframes blinker {
                                50% { opacity: 0; }
                            }
                            </style>
                            """, 
                            unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                            """
                            <div style="text-align: center; animation: blinker 1.5s linear infinite; color: red; font-weight: bold;">
                                ⚠️ Open AI response doesn`t match the database answer ⚠️
                            </div>
                            <style>
                            @keyframes blinker {
                                50% { opacity: 0; }
                            }
                            </style>
                            """, 
                            unsafe_allow_html=True
                            )
                except Exception as e:
                    st.error(f"An error occurred while regenerating the answer: {str(e)}")

            if st.button("Go Home", key="go_home_button_alt"):
                st.session_state.page = 'main'
                st.rerun()
                
def visualizations_page():

    # Custom CSS for black background and white text
    st.markdown("""
        <style>
        body {
            background-color: #FFFFFF;
            color: black;
        }
        .stSelectbox label {
            font-size: 2.2em !important;
            color: white !important;
        }
        .stTextInput>div>div>input, .stSelectbox>div>div>div>input {
            color: black;
        }
        h1 {
            font-family: "Times New Roman", Arial, sans-serif;
            text-align: center;
            font-size: 2.75em;
            color: #000000;
            padding-top: 1em;
            padding-bottom: 0.5em;
        }
        .stSelectbox, .stDataFrame {
            background-color: #222222;
            border-radius: 10px;
        }
        .stMarkdown {
            color: black;
        }
        .stButton button {
            background-color: #000000;
            color: white;
        }
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
        }
        .main, .block-container {
            background-color: #FFFFFF; 
        }
        .stMarkdown h2 {
            color: #000000;
        }
        
        label {
            font-size: 1.25em; /* Adjust the font size (4 points larger) */
            color: black;      /* White color for the label */
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Visualizations")
    data = get_visualization_data()

    st.sidebar.title("Filter")
    all_tasks = data['Task_No'].unique().tolist()

    # Add "Select All" option
    select_all = st.sidebar.checkbox("Select All")

    if not select_all:
        # Text input for task numbers
        task_input = st.sidebar.text_input("Enter Task Numbers (comma-separated)", 
                                           placeholder="e.g., 1,2,3-5,7")
        
        if task_input:
            # Parse the input
            selected_tasks = []
            for part in task_input.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected_tasks.extend(range(start, end + 1))
                else:
                    selected_tasks.append(int(part.strip()))
            
            selected_tasks = [task for task in selected_tasks if task in all_tasks]
        else:
            selected_tasks = []
    else:
        selected_tasks = all_tasks

    # Filter data based on selection
    filtered_data = data[data['Task_No'].isin(selected_tasks)]

    # Display selected task IDs
    if not select_all:
        st.sidebar.write("Selected Task IDs:")
        for task in selected_tasks:
            task_id = data[data['Task_No'] == task]['task_id'].iloc[0]
            st.sidebar.write(f"Task {task} (ID: {task_id})")

    # Center content with margins
    st.markdown("""
        <style>
            .reportview-container .main .block-container {
                max-width: 1000px;
                padding-top: 7rem;
                padding-right: 4in;
                padding-left: 4in;
                padding-bottom: 7rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # Visualizations
    st.subheader("1. Number of Attempts vs Count of Task IDs")
    attempts_count_chart(filtered_data)

    st.subheader("2. Correct Responses: First Attempt vs Regenerated Attempts")
    correct_responses_chart(filtered_data)

    st.subheader("3. Validation Status of Task IDs")
    validation_status_chart(filtered_data)

    if st.button("Back to Home Page"):
        st.session_state.page = 'main'
        st.rerun()

if __name__ == "__main__":
    main()