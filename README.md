# Task Genie: AI-Powered Answer Validation System

This project involves developing a Streamlit application for model evaluation teams to select and evaluate test cases from the GAIA dataset against the OpenAI model. The application allows users to send context data and questions to the OpenAI model and compare the model's answers with the final answers from the metadata file.

Codelab link - [link](https://codelabs-preview.appspot.com/?file_id=11Bv8yal4awS5ywv5cM-QZX0SG4z43PxAG1WWGQWo7o0#0)

### Attestation

WE ATTEST THAT WE HAVEN’T USED ANY OTHER STUDENTS’ WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK.

**Contribution:**
- Vaishnavi Veerkumar: 34%
- Sriram Venkatesh: 33%
- Siddharth Pawar: 33%

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Application Workflow](#application-workflow)
- [Data Flow and Backend Processes](#data-flow-and-backend-processes)
- [Usage Instructions](#usage-instructions)
- [Visualizations](#visualizations)
- [Challenges & Solutions](#challenges--solutions)
- [Video Link](#video-link)
- [License](#license)

## Project Overview

This application automates task validation based on predefined criteria stored in a MySQL database. It leverages OpenAI's API to generate answers for given tasks and compares them with the expected answers stored in the database using **cosine similarity**. The app also tracks the number of attempts made and visualizes task progress through bar charts and pie charts for better understanding of validation performance.

### Features:
- Task selection and submission via Streamlit interface.
- Fetch answers from **AWS S3** and **MySQL**.
- Cosine similarity-based validation of answers using **Sentence Transformers**.
- Tracks number of attempts per task.
- Various visualizations for performance analytics (correct attempts, validation status, etc.).

## Technologies Used

- **Streamlit**: Frontend framework
- **MySQL**: Backend database for task metadata and answers
- **AWS S3**: Storage for external files (audio, images, PDFs)
- **OpenAI API**: For generating task responses
- **Sentence Transformers**: For embedding and comparing sentences
- **Matplotlib & Seaborn**: For data visualizations
- **Boto3**: AWS S3 integration
- **PyPDF2**: For PDF file reading
- **Pandas**: Data handling and display

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/task-genie.git
   cd task-genie
2. **Setup Virtual Env**
    ```bash
    1.  python -m venv venvsource venv/bin/activate
    
    2.  \# On Windows use \`venv\\Scripts\\activate\`
    
    3.  pip install -r requirements.txt
    
3.  **MySQL Setup**:
    
    *   Ensure you have MySQL installed and running.
        
    *   Create the GAIA database and populate it with task metadata (see the /db folder for SQL schema).
        
4.  **AWS S3 Setup**:
    
    *   Make sure your AWS credentials (aws\_access\_key\_id, aws\_secret\_access\_key) are set correctly to access the S3 bucket containing the task files.
        
6.  **OpenAI API Key**:
    
    *   Update the openai.api\_key in the script with your OpenAI API key.
        

Application Workflow
--------------------

### Step 1: Streamlit Display

*   On the homepage, the user selects a task from a dropdown that fetches questions stored in the **MySQL** database.
    

### Step 2: Generate Answers

*   Once a task is selected, the application uses the **OpenAI API** to generate an answer for the selected question.
    

### Step 3: Cosine Similarity Check

*   The generated answer is compared with the correct answer from the database using cosine similarity. If the similarity is above a predefined threshold, the answer is validated.
    

### Step 4: Regenerate Answers

*   If the initial answer doesn't meet the threshold, the user can attempt to regenerate the answer. The number of attempts is tracked and updated in the MySQL database.
    

### Step 5: Update Attempts & Validation

*   The application updates the number of attempts and the validation status (whether the answer matches the correct one) in the MySQL database.
    

### Step 6: Visualizations

*   A separate page displays various visualizations such as:
    
    *   **Number of Attempts per Task** (Bar Chart)
        
    *   **Correct Responses: First Attempt vs Regenerated Attempts** (Bar Chart)
        
    *   **Validation Status of Task IDs** (Pie Chart)
        

Data Flow and Backend Processes
-------------------------------

1.  **Frontend Input**: Users interact with the app through Streamlit's UI, selecting tasks and generating answers.
    
2.  **Data Storage**:
    
    *   Task metadata (questions, correct answers, validation status, etc.) is stored in a **MySQL** database.
        
    *   Files such as PDFs, images, and audios related to tasks are stored in **AWS S3**.
        
3.  **Processing**:
    
    *   The application generates answers using **OpenAI** and compares them with the correct answer using **cosine similarity** from the **Sentence Transformers** library.
        
4.  **Output**:
    
    *   The result (validated or not) is stored back into MySQL.
        
    *   The user can view the validation status, number of attempts, and other metrics through data visualizations.
        

Usage Instructions
------------------

streamlit run main.py

1.  **Task Selection**:
    
    *   Select a task from the dropdown to see the question.
        
2.  **Answer Generation**:
    
    *   Click on the "Generate Answer" button to receive an AI-generated answer from OpenAI.
        
3.  **Regenerate Answer**:
    
    *   If the initial answer is incorrect, click "Regenerate" to try again.
        
4.  **View Visualizations**:
    
    *   Visit the Visualizations page to see analytics on the number of attempts, validation status, etc.
        

Visualizations
--------------

1.  **Number of Attempts per Task**:Displays a horizontal bar chart showing the number of attempts taken for each task.
    
2.  **Correct Responses: First Attempt vs Regenerated Attempts**:Bar chart comparing the number of correct answers on the first attempt versus those that required regeneration.
    
3.  **Validation Status of Task IDs**:Pie chart showing the percentage of tasks that have been validated versus those that haven’t.
    

Challenges & Solutions
----------------------

*   **Data Validation**: One challenge was ensuring the AI-generated responses were evaluated correctly. To solve this, we implemented cosine similarity on pre-trained embeddings and compared them with stored answers.
    
*   **File Handling**: Managing multiple file types (audio, PDFs, images) from AWS S3 required additional parsing and formatting, which was solved by using libraries like PyPDF2 and PIL.

    
Video Link
----------------------
https://youtu.be/kUk6Q0LkTsU


License
-------

This project is licensed under the MIT License. See the LICENSE file for more details.

