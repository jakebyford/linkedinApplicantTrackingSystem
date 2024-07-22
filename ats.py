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

# Download NLTK stopwords
nltk.download('stopwords')


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

            # Get top important words
            my_top_words = get_top_tfidf_words(preprocessed_resume)
            jd_top_words = get_top_tfidf_words(preprocessed_job_description)

            # Include top_words in the similarity_scores dictionary
            similarity_scores[filename] = {
                'score': similarity_score, 'jd_top_words': jd_top_words, 'my_top_words': my_top_words}

    sorted_similarity_scores = sorted(
        similarity_scores.items(), key=lambda x: x[1]['score'], reverse=True)

    for company, data in sorted_similarity_scores[:10]:
        score = data['score']
        jd_top_words = data['jd_top_words']
        my_top_words = data['my_top_words']

        # if score >= 0.3:
        print("################################################")
        print(f"JOB: \n{company}")
        print("")
        print(
            f"RESUME VS. JOB DESCRIPTION: \n{round(score, 2)}% match")
        print("")
        print(f"WORD IMPORTANCE RANKING ON JOB DESCRIPTION:")
        for i in jd_top_words:
            print(f"{i[0]}: {round(i[1] * 100, 2)}%")
        print("")
        print(f"WORD IMPORTANCE RANKING ON MY RESUME:")
        for i in my_top_words:
            print(f"{i[0]}: {round(i[1] * 100, 2)}%")

    # TODO: Additional steps for machine learning model
    # ...


if __name__ == "__main__":
    main()
