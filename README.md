# Task Genie: AI-Powered Answer Validation System

This project involves developing a Streamlit application for model evaluation teams to select and evaluate test cases from the GAIA dataset against the OpenAI model. The application allows users to send context data and questions to the OpenAI model and compare the model's answers with the final answers from the metadata file.

[![Codelabs](https://img.shields.io/badge/Codelabs-green?style=for-the-badge)](https://codelabs-preview.appspot.com/?file_id=11Bv8yal4awS5ywv5cM-QZX0SG4z43PxAG1WWGQWo7o0#0)

Video Link
----------------------
https://youtu.be/kUk6Q0LkTsU

### Attestation

WE ATTEST THAT WE HAVEN'T USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK

## Resources and Contribution

| Group Member        | Contribution (%) |
|---------------------|------------------|
| Siddharth Pawar     | 33.33%          |
| Vaishnavi Veerkumar | 33.33%          |
| Sriram Venkatesh    | 33.33%          |



## Project Overview

This application automates task validation by leveraging predefined criteria stored in a MySQL database. It incorporates a robust system to retrieve and store metadata from Hugging Face into a PostgreSQL database, while associated unstructured data is uploaded to AWS S3 for seamless integration.
A Streamlit-based frontend enables intuitive task exploration and OpenAI-powered querying, utilizing OpenAI's API to generate responses for given tasks and comparing them with the expected answers using advanced validation techniques. Cosine similarity is implemented with Sentence Transformers to assess answer accuracy, providing a reliable validation mechanism.
Additionally, the application visualizes OpenAI query attempts and efficiency insights within Streamlit, using bar charts and pie charts to track task progress and performance metrics. It monitors the number of query attempts, offering clear insights into validation performance for enhanced decision-making.

## Technologies Used

| **Technology/Tool**     | **Purpose**                                                  |
|--------------------------|--------------------------------------------------------------|
| **Streamlit**            | Frontend framework for building interactive applications.    |
| **MySQL DB**             | Backend for managing task IDs, task metadata, annotator steps, and database answers. |
| **AWS S3**               | Storage solution for external files such as audio, images, and PDFs. |
| **OpenAI API**           | Generating task responses based on inputs.                  |
| **Sentence Transformers**| Generating embeddings and comparing sentences for analysis. |
| **Matplotlib & Seaborn** | Creating data visualizations and graphical representations.  |
| **Boto3**                | Integrating with AWS S3 for file upload and retrieval.       |
| **PyPDF2**               | Reading and extracting data from PDF files.                 |
| **Pandas**               | Handling, processing, and displaying structured data.       |


## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/TechGenie_AI_Powered_Answers.git
   cd TechGenie_AI_Powered_Answers
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
    
    *   Update the openai_api_key in the script with your OpenAI API key.
        
## Environment Variables

Create a `.env` file with the following variables:

### API Keys & AWS Setup

```bash
OPENAI_API_KEY=your_openai_api_key
AWS_BUCKET_NAME=your_bucket_name
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_scret_access_key
AWS_REGION=your_aws_region
```

## Application Workflow
--------------------

Step 1: Streamlit Display
*   On the homepage, the user selects a task from a dropdown that fetches questions stored in the **MySQL** database.

Step 2: Generate Answers
*   Once a task is selected, the application uses the **OpenAI API** to generate an answer for the selected question.
    
Step 3: Cosine Similarity Check
*   The generated answer is compared with the correct answer from the database using cosine similarity. If the similarity is above a predefined threshold, the answer is validated.

Step 4: Regenerate Answers
*   If the initial answer doesn't meet the threshold, the user can attempt to regenerate the answer. The number of attempts is tracked and updated in the MySQL database.
    
Step 5: Update Attempts & Validation
*   The application updates the number of attempts and the validation status (whether the answer matches the correct one) in the MySQL database.
    
Step 6: Visualizations
*   A separate page displays various visualizations such as:
    
Number of Attempts per Task (Bar Chart)

Correct Responses: First Attempt vs Regenerated Attempts (Bar Chart)

Validation Status of Task IDs (Pie Chart)
        

    

## Challenges & Solutions
----------------------

*   **Data Validation**: One challenge was ensuring the AI-generated responses were evaluated correctly. To solve this, we implemented cosine similarity on pre-trained embeddings and compared them with stored answers.
    
*   **File Handling**: Managing multiple file types (audio, PDFs, images) from AWS S3 required additional parsing and formatting, which was solved by using libraries like PyPDF2 and PIL.

    



## License
-------

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

