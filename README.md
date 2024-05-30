# 24PDInsight - Bizzagi Company Capstone Project

## Introduction
Welcome to **24PDInsight**, the capstone project developed by *Bangkit Cohort 2024 Batch 1* for the **Bizzagi** company. This project aims to provide insights and analysis for the capstone company project undertaken as part of the Bangkit program. 

## Features
- [x] **Transcribe Audio to Text:** Utilize Whisper from OpenAI to transcribe audio files to text.
- [x] **Wordcloud Generation:** Calculate the frequency of words in the text data and pass it to the frontend for generating wordcloud visualizations.
- [x] **Sentiment Analysis:** Analyze the sentiment of text data to determine the overall sentiment (positive, negative, or neutral).
- [ ] **Topic Modeling:** Identify topics in text data and analyze their distribution.
- [ ] **Named Entity Recognition (NER):** Identify and classify named entities (e.g., persons, organizations, locations) in text data.
- [x] **Text Summarization:** Automatically generate concise summaries of text documents.

## Getting Started
To get started with 24PDInsight, follow these steps:

1. Clone the repository to your local machine.
    ```bash
    git clone https://github.com/your-username/24PDInsight.git
    ```

2. Create and Activate a Virtual Environment:
    ```bash
    python3.10 -m venv env 
    source env/bin/activate  
    # On Windows, use "env\Scripts\activate"
    ```

3. Install the required dependencies listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Application:
    ```bash
    python app.py
    ```
## Testing
We have included a Thunder Client collection for testing the API endpoints. You can find the collection file `thunder-collection_24PDInsight.json` in the repository. To use this file:

1. Install the Thunder Client extension in Visual Studio Code.
2. Open Thunder Client and import the `thunder-collection_24PDInsight.json` file.
3. Run the requests to test the different features of the application.