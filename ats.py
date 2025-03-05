import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from config import my_path
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')

# Define reusable template
resume_template = PromptTemplate.from_template("""
    Improve the resume bullet points to match the job description while keeping them realistic and professional.
    Provide 3 optimized bullet points.

    Job Description:
    {job_description}

    Resume Excerpt:
    {resume_text}

    Improved Bullet Points:
""")

# Define reusable prompt template
cover_letter_template = PromptTemplate.from_template("""
    Write a compelling, one-page cover letter tailored to the job description below.
    Ensure it includes a strong opening, highlights key skills, and ends with a confident closing.

    Job Description:
    {job_description}

    Resume Excerpt:
    {resume_text}

    Cover Letter:
""")

# Define reusable AI model
# model = ChatOpenAI(model="gpt-4")
model = ChatOpenAI(model="mistralai/Mistral-7B-Instruct-v0.2")


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Parameters:
    - pdf_path (str): Path to the PDF file.

    Returns:
    - str: Extracted text from the PDF.
    """
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text


def extract_text_from_txt(txt_path):
    """
    Extract text from a text file.

    Parameters:
    - txt_path (str): Path to the text file.

    Returns:
    - str: Extracted text from the file.
    """
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def extract_text_from_docx(docx_path):
    """
    Extract text from a Word document (Docx).

    Parameters:
    - docx_path (str): Path to the Word document (Docx).

    Returns:
    - str: Extracted text from the document.
    """
    return docx2txt.process(docx_path)


def process_all_text_files(folder_path):
    """
    Process all text files in a given folder.

    Parameters:
    - folder_path (str): Path to the folder containing text files.

    Returns:
    - list: List of extracted texts from all text files.
    """
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_txt(file_path)
            texts.append(text)
    return texts


def preprocess_text(text):
    """
    Preprocess the input text by converting to lowercase,
    tokenizing, removing non-alphanumeric tokens, and filtering out stopwords.

    Parameters:
    - text (str): Input text.

    Returns:
    - str: Preprocessed text.
    """
    if not text:
        return ''  # Return an empty string if input text is empty

    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [
        word for word in tokens if word.isalnum() and word not in stop_words]

    preprocessed_text = ' '.join(filtered_tokens)
    # Return empty string if all tokens are removed
    return preprocessed_text if preprocessed_text else ''


def map_similarity_score(similarity, target_range=(50, 100)):
    """
    Map the similarity score to a specified target range.

    Parameters:
    - similarity (float): Original similarity score in the range of 0-1.
    - target_range (tuple): Target range for the mapped similarity score.

    Returns:
    - float: Mapped similarity score within the target range.

    Example:
    - similarity = 0.35
    - mapped_score = 50 + (0.35 - 0) * (100 - 50) / (1 - 0) = 50 + 17.5 = 67.5
    """
    original_range = (0, 1)
    mapped_score = target_range[0] + (similarity - original_range[0]) * (
        target_range[1] - target_range[0]
    ) / (original_range[1] - original_range[0])
    return mapped_score


def calculate_similarity(resume_text, job_description_text):
    """
    Calculate the cosine similarity between two text strings.

    Parameters:
    - resume_text (str): Text from the resume.
    - job_description_text (str): Text from the job description.

    Returns:
    - float: Mapped cosine similarity score within the target range.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    mapped_similarity = map_similarity_score(
        similarity, target_range=(50, 100))
    return mapped_similarity


# Load SBERT model (only needs to be done once)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity_sbert(resume_text, job_description_text):
    """
    Calculate the similarity between resume and job description using SBERT.

    Parameters:
    - resume_text (str): Preprocessed resume text.
    - job_description_text (str): Preprocessed job description text.

    Returns:
    - float: Mapped similarity score within the target range.
    """
    resume_embedding = sbert_model.encode(resume_text)
    job_embedding = sbert_model.encode(job_description_text)

    similarity = np.dot(resume_embedding, job_embedding) / (
        np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
    )

    mapped_similarity = map_similarity_score(similarity, target_range=(50, 100))
    return mapped_similarity


def get_top_tfidf_words(text, top_n=5):
    """
    Get the top N important words based on TF-IDF scores.

    Parameters:
    - text (str): Input text.
    - top_n (int): Number of top words to retrieve.

    Returns:
    - list: List of tuples (word, TF-IDF score) for the top words.
    """
    if not text:
        return []

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()

    if not feature_names.any():  # Use .any() instead of a direct boolean check
        return []

    tfidf_scores = zip(feature_names, vectors.toarray()[0])
    sorted_tfidf_scores = sorted(
        tfidf_scores, key=lambda x: x[1], reverse=True)

    return sorted_tfidf_scores[:top_n]


def find_missing_keywords(job_description_text, resume_text, top_n=10):
    """
    Identify important job description keywords that are missing from the resume.

    Parameters:
    - job_description_text (str): The job description text.
    - resume_text (str): The resume text.
    - top_n (int): Number of top keywords to consider from the job description.

    Returns:
    - list: Keywords that are important but missing in the resume.
    """
    job_tfidf = get_top_tfidf_words(job_description_text, top_n)
    resume_tfidf = get_top_tfidf_words(resume_text, top_n)

    job_keywords = {word for word, _ in job_tfidf}
    resume_keywords = {word for word, _ in resume_tfidf}

    missing_keywords = job_keywords - resume_keywords  # Find what's missing
    return list(missing_keywords)


def improve_resume_bullet_points(resume_text, job_description):
    """
    Enhances resume bullet points to align better with a job description.

    Parameters:
    - resume_text (str): Original resume text.
    - job_description (str): Job description text.

    Returns:
    - list: List of AI-improved bullet points.
    """
    
    # Create prompt
    prompt = resume_template.invoke({"job_description": job_description, "resume_text": resume_text})
    
    # Get AI-generated response
    response = model.invoke(prompt)
    
    # Process response
    improved_bullets = response.content.split("\n")
    
    return [bullet.strip() for bullet in improved_bullets if bullet.strip()]


def generate_cover_letter(job_description, resume_text):
    """
    Generates a tailored cover letter based on a job description and resume.

    Parameters:
    - job_description (str): Job description text.
    - resume_text (str): Resume text.

    Returns:
    - str: AI-generated cover letter.
    """
    
    # Create prompt using the template
    prompt = cover_letter_template.invoke({"job_description": job_description, "resume_text": resume_text})
    
    # Get AI-generated response
    response = model.invoke(prompt)
    
    return response.content.strip()



def main():
    """
    Main function to prompt the user for a PDF resume and a job description,
    then calculate and print the similarity score.
    """
    # Input: PDF resume
    Tk().withdraw()  # Hides main window in tkinter app
    pdf_resume_path = askopenfilename(title="Select PDF Resume")

    # Extract text from PDF resume
    resume_text = extract_text_from_pdf(pdf_resume_path)

    # Preprocess resume
    preprocessed_resume = preprocess_text(resume_text)

    folder_path = my_path

    # processed_job_descriptions = process_all_text_files(folder_path)
    similarity_scores = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            # Input: Job description (copied and pasted)
            job_description = extract_text_from_txt(file_path)

            # Preprocess text
            preprocessed_job_description = preprocess_text(job_description)
            # print(preprocessed_job_description)

            # Calculate similarity
            similarity_score = round(calculate_similarity(
                preprocessed_resume, preprocessed_job_description), 2)
            
            # Calculate similarity using SBERT
            sbert_similarity_score = round(calculate_similarity_sbert(preprocessed_resume, preprocessed_job_description), 2)

            # Get top important words
            my_top_words = get_top_tfidf_words(preprocessed_resume)
            jd_top_words = get_top_tfidf_words(preprocessed_job_description)
            
            # Find missing keywords
            missing_keywords = find_missing_keywords(preprocessed_job_description, preprocessed_resume)

            # Include top_words in the similarity_scores dictionary
            similarity_scores[filename] = {
                'score': similarity_score, 
                'sbert_score': sbert_similarity_score, 
                'jd_top_words': jd_top_words, 
                'my_top_words': my_top_words, 
                'missing_keywords': missing_keywords
                }    
    
    sorted_similarity_scores = sorted(
        similarity_scores.items(), key=lambda x: x[1]['sbert_score'], reverse=True)
    
    print("Found Top Matches")
    top_matches = sorted_similarity_scores[:5]  # Get top 5 matching companies

    results = {}  # Store improved bullets & cover letters
    print("Creating AI-Suggested Resume Improvements and Cover Letters")
    for company, score in top_matches:
        file_path = os.path.join(folder_path, company)

        if os.path.exists(file_path):  # Check if the file exists
            # Extract and preprocess job description
            job_description = extract_text_from_txt(file_path)
            preprocessed_job_description = preprocess_text(job_description)

            # Improve resume bullets
            improved_bullets = improve_resume_bullet_points(preprocessed_resume, preprocessed_job_description)

            # Generate AI cover letter
            cover_letter = generate_cover_letter(preprocessed_job_description, preprocessed_resume)

            # Store results
            results[company] = {
                "score": score,
                "improved_bullets": improved_bullets,
                "cover_letter": cover_letter
            }

    # for company, data in sorted_similarity_scores[:5]:
    #     score = data['score']
    #     sbert_score = data['sbert_score']
    #     jd_top_words = data['jd_top_words']
    #     my_top_words = data['my_top_words']
    #     my_missing_keywords = data['missing_keywords']

        # if score >= 0.3:
        # print("################################################")
        # print(f"JOB: \n{company}")
        # print("")
        # print(
        #     f"RESUME VS. JOB DESCRIPTION: \n{round(score, 2)}% match")
        # print("")
        # print(
        #     f"RESUME VS. JOB DESCRIPTION (SBERT): \n{round(sbert_score, 2)}% match")
        # print("")
        # print(f"WORD IMPORTANCE RANKING ON JOB DESCRIPTION:")
        # for i in jd_top_words:
        #     print(f"{i[0]}: {round(i[1] * 100, 2)}%")
        # print("")
        # print(f"WORD IMPORTANCE RANKING ON MY RESUME:")
        # for i in my_top_words:
        #     print(f"{i[0]}: {round(i[1] * 100, 2)}%")
        # print("")

        # print(f"🚀 Missing Keywords to Add:")
        # if my_missing_keywords:
        #     print(", ".join(my_missing_keywords))
        # else:
        #     print("✅ No critical keywords missing!")

        # print("################################################")
        # print(f"JOB: {filename}")
        # print(f"RESUME VS. JOB DESCRIPTION: {sbert_score}% match")
        

    for company, data in results.items():
        sbert_score = data.get('score', 0)
        improved_bullets = data.get('improved_bullets', [])
        cover_letter = data.get('cover_letter', "No cover letter available")

        print("################################################")
        print(f"JOB: {company}")  
        print(f"RESUME VS. JOB DESCRIPTION: {sbert_score}% match")

        print("\n🚀 AI-Suggested Resume Improvements:")
        for bullet in improved_bullets:
            print(f"- {bullet}")

        print("\n📄 AI-Generated Cover Letter:")
        print(cover_letter)


    # TODO: Additional steps for machine learning model
    # ...


if __name__ == "__main__":
    main()
