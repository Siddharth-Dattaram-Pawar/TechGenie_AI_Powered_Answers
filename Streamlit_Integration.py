import streamlit as st
import pymysql
import boto3
import pandas as pd
from io import BytesIO
from PIL import Image
import zipfile
from PyPDF2 import PdfReader

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

# JavaScript function to set query parameters
def set_query_params(page):
    st.write(f"""
        <script>
            const queryParams = new URLSearchParams(window.location.search);
            queryParams.set("page", "{page}");
            window.history.replaceState(null, null, "?" + queryParams.toString());
        </script>
    """, unsafe_allow_html=True)

# Navigation functions
def go_to_page(page):
    set_query_params(page)

def get_current_page():
    query_params = st.query_params
    return query_params.get('page', ['main'])[0]

# Streamlit main page
def main_page():
    st.markdown("""
        <style>
        body {
            background-color: #121212;
            color: white;
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

                # Display or provide preview for the file
                display_file_from_s3(file_name)
        else:
            # Flashing and centered message
            st.markdown(
                """
                <div style="text-align: center; animation: blinker 1.5s linear infinite; color: red; font-weight: bold;">
                    ⚠️ No additional content found for this question ⚠️
                </div>
                <style>
                @keyframes blinker {
                    50% { opacity: 0; }
                }
                </style>
                """, 
                unsafe_allow_html=True
            )

    # Add buttons for navigation
    if st.button("Go to New Page"):
        go_to_page('new_page')

def new_page():
    st.title("New Page")
    st.write("This is a new page!")

    # Button to go back to main page
    if st.button("Go Back"):
        go_to_page('main')

# Main function to control navigation
def main():
    current_page = get_current_page()

    if current_page == 'main':
        main_page()
    elif current_page == 'new_page':
        new_page()

if __name__ == "__main__":
    main()