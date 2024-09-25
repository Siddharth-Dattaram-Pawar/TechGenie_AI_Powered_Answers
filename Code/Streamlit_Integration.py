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
    query = "SELECT task_no, Number_of_Attempts, Validated FROM METADATA ORDER BY task_no"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    
    # Create a DataFrame
    df = pd.DataFrame(result, columns=['Task_No', 'Number_of_Attempts', 'Validated'])
    return df

def compare_answers(openai_answer, database_answer, task_id, threshold=0.8):
    # Check if database answer is a single word and if it appears in the generated answer
    database_answer_cleaned = clean_text(database_answer)
    if len(database_answer_cleaned.split()) == 1:
        if database_answer_cleaned in clean_text(openai_answer):
            update_validation(task_id, 'Yes')
            st.write("Exact match found for single-word database answer.")
            return True
    
    # Fall back to cosine similarity if no direct match was found
    similarity = cosine_similarity_embeddings(openai_answer, database_answer)
    if similarity >= threshold:
        update_validation(task_id, 'Yes')
        return True
    
    # If no match
    update_validation(task_id, 'No')
    return False

def decreasing_bar_chart(data):
    df = data.sort_values('Number_of_Attempts', ascending=True)
    
    plt.figure(figsize=(10, max(8, len(df) * 0.3)))  # Adjust figure height based on number of tasks
    bars = plt.barh(df['Task_No'], df['Number_of_Attempts'])
    plt.title('Number of Attempts per Task')
    plt.xlabel('Number of Attempts')
    plt.ylabel('Task Number')
    plt.yticks(df['Task_No'])
    
    # Add value labels to the end of each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width}', 
                 ha='left', va='center', fontweight='bold')
    
    st.pyplot(plt)

def correct_responses_chart(data):
    df = pd.DataFrame(data, columns=['task_id', 'Number_of_Attempts', 'Validated'])
    first_attempt = df[(df['Number_of_Attempts'] == 1) & (df['Validated'] == 'Yes')].shape[0]
    regenerated_attempts = df[(df['Number_of_Attempts'] > 1) & (df['Validated'] == 'Yes')].shape[0]
    
    plt.figure(figsize=(6, 4))
    plt.bar(['First Attempt', 'Regenerated Attempts'], [first_attempt, regenerated_attempts])
    plt.title('Correct Responses: First Attempt vs Regenerated Attempts')
    plt.ylabel('Count')
    for i, v in enumerate([first_attempt, regenerated_attempts]):
        plt.text(i, v, str(v), ha='center', va='bottom')
    st.pyplot(plt)

def validation_status_chart(data):
    df = pd.DataFrame(data, columns=['task_id', 'Number_of_Attempts', 'Validated'])
    validated_count = df[df['Validated'] == 'Yes'].shape[0]
    not_validated_count = df[df['Validated'] == 'No'].shape[0]
    
    plt.figure(figsize=(6, 6))
    plt.pie([validated_count, not_validated_count], labels=['Validated', 'Not Validated'], autopct='%1.1f%%')
    plt.title('Validation Status of Task IDs')
    st.pyplot(plt)

def visualizations_page():
    st.title("Visualizations")
    
    data = get_visualization_data()
    
    st.subheader("1. Number of Attempts per Task")
    decreasing_bar_chart(data)
    
    st.subheader("2. Correct Responses: First Attempt vs Regenerated Attempts")
    correct_responses_chart(data)
    
    st.subheader("3. Validation Status of Task IDs")
    validation_status_chart(data)
    
    if st.button("Back to Home Page"):
        st.session_state.page = 'main'
        st.rerun()
        
# Streamlit main page
def main():
    st.set_page_config(page_title="Task Genie: AI-Powered Answers", layout="wide")

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
    st.title("Open AI Response Comparison")
    
    if 'selected_question' in st.session_state:
        st.write(f"**Selected Question:** {st.session_state.selected_question}")
        st.write(f"**OpenAI Answer:** {st.session_state.openai_answer}")
        st.write(f"**Database Answer:** {st.session_state.database_answer}")

        if st.session_state.is_correct:
            st.success("The OpenAI answer matches the database answer.")
        else:
            st.error("The OpenAI answer does not match the database answer.")

    if st.button("Regenerate Response"):
        update_attempts(st.session_state.selected_task_id)  # Increment count here
        st.session_state.page = 'regenerate_page'
        st.rerun()
    
    # Navigation buttons
    if st.button("Go Back"):
        st.session_state.page = 'main'
        st.rerun()

    

def regenerate_answer_page():
    st.title("Regenerated Response")

    # Ensure the question is available in session state
    if 'selected_question' in st.session_state:
        st.write(f"**Selected Question:** {st.session_state.selected_question}")

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
                st.write(f"**Task ID:** {st.session_state.selected_task_id}")

                # Display task details, annotator steps, and files
                st.write(f"**Annotator Steps:** {annotator_steps}")
                if 'associated_files' in st.session_state:
                    st.write(f"**Associated Files:** {st.session_state.associated_files}")
                st.write("**OpenAI Generated Answer:**")
                st.write(openai_answer)
                
                # Compare the OpenAI answer with the database answer
                if 'database_answer' in st.session_state:
                    is_correct = compare_answers(openai_answer, st.session_state.database_answer, st.session_state.selected_task_id)
                    st.session_state.is_correct = is_correct

                    if is_correct:
                        st.success("The generated OpenAI answer matches the database answer.")
                    else:
                        st.error("The generated OpenAI answer does not match the database answer.")
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
                            st.success("The regenerated OpenAI answer matches the database answer.")
                        else:
                            st.error("The regenerated OpenAI answer does not match the database answer.")
                except Exception as e:
                    st.error(f"An error occurred while regenerating the answer: {str(e)}")

            if st.button("Go Home", key="go_home_button_alt"):
                st.session_state.page = 'main'
                st.rerun()

if __name__ == "__main__":
    main()
