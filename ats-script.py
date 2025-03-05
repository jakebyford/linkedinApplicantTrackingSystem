import os
import numpy as np
import nltk
import PyPDF2
import docx2txt
import torch
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from huggingface_hub import InferenceClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import my_path


# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126


# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)  # Fixed 'punkt_tab' to 'punkt'

class HuggingFaceModel:
    """A wrapper for Hugging Face Inference API to provide an interface similar to LangChain."""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        """Initialize a Hugging Face Inference API client.
        
        Args:
            model_name: The Hugging Face model to use
        """
        self.model_name = model_name
        self.client = InferenceClient()
        
    def invoke(self, prompt):
        """Generate text based on the prompt using HF Inference API.
        
        Args:
            prompt: A string or prompt template object with content property
        
        Returns:
            An object with a content property containing the generated text
        """
        # Handle both string prompts and template results
        prompt_text = prompt.content if hasattr(prompt, 'content') else prompt
        
        # Set up messages for chat completion format
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt_text}
        ]
        
        # Make the API call
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract the generated text
        response_text = response['choices'][0]['message']['content']
        
        # Create a response object with a content property to match the existing interface
        class Response:
            def __init__(self, content):
                self.content = content
                
        return Response(response_text)


class SimplePromptTemplate:
    """A simplified version of LangChain's PromptTemplate."""
    
    def __init__(self, template):
        """Initialize a simple prompt template.
        
        Args:
            template: The template string with {placeholders}
        """
        self.template = template
        
    def invoke(self, values):
        """Fill the template with the provided values.
        
        Args:
            values: Dictionary of values to fill the template
            
        Returns:
            The filled template string
        """
        return self.template.format(**values)


# --- TEXT EXTRACTION FUNCTIONS ---

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''.join(page.extract_text() for page in pdf_reader.pages)
    return text


def extract_text_from_txt(txt_path):
    """Extract text from a text file."""
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()


def extract_text_from_docx(docx_path):
    """Extract text from a Word document (Docx)."""
    return docx2txt.process(docx_path)


def extract_text(file_path):
    """Extract text from a file based on its extension."""
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


# --- TEXT PROCESSING FUNCTIONS ---

def preprocess_text(text):
    """Preprocess text by tokenizing and removing stopwords."""
    if not text:
        return ''

    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    return ' '.join(filtered_tokens) or ''


def get_top_tfidf_words(text, top_n=5):
    """Get the top N important words based on TF-IDF scores."""
    if not text:
        return []

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()

    if not feature_names.any():
        return []

    tfidf_scores = zip(feature_names, vectors.toarray()[0])
    return sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_n]


def find_missing_keywords(job_description_text, resume_text, top_n=10):
    """Identify important job description keywords missing from the resume."""
    job_tfidf = get_top_tfidf_words(job_description_text, top_n)
    resume_tfidf = get_top_tfidf_words(resume_text, top_n)

    job_keywords = {word for word, _ in job_tfidf}
    resume_keywords = {word for word, _ in resume_tfidf}

    return list(job_keywords - resume_keywords)


# --- SIMILARITY CALCULATION FUNCTIONS ---

def map_similarity_score(similarity, target_range=(50, 100)):
    """Map the similarity score to a specified target range."""
    original_range = (0, 1)
    mapped_score = target_range[0] + (similarity - original_range[0]) * (
        target_range[1] - target_range[0]
    ) / (original_range[1] - original_range[0])
    return mapped_score


def calculate_tfidf_similarity(resume_text, job_description_text):
    """Calculate cosine similarity between resume and job description using TF-IDF."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return map_similarity_score(similarity)


def calculate_sbert_similarity(resume_text, job_description_text, model):
    """Calculate similarity using SBERT embeddings."""
    resume_embedding = model.encode(resume_text)
    job_embedding = model.encode(job_description_text)

    similarity = np.dot(resume_embedding, job_embedding) / (
        np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
    )

    return map_similarity_score(similarity)


# --- AI ENHANCEMENT FUNCTIONS ---

def improve_resume_bullet_points(resume_text, job_description, model, template):
    """Enhance resume bullet points to align better with a job description."""
    prompt = template.invoke({
        "job_description": job_description, 
        "resume_text": resume_text
    })
    
    response = model.invoke(prompt)
    improved_bullets = response.content.split("\n")
    
    return [bullet.strip() for bullet in improved_bullets if bullet.strip()]


def generate_cover_letter(job_description, resume_text, model, template):
    """Generate a tailored cover letter based on job description and resume."""
    prompt = template.invoke({
        "job_description": job_description, 
        "resume_text": resume_text
    })
    
    response = model.invoke(prompt)
    return response.content.strip()


# --- MAIN FUNCTION ---

def main():
    """Main function to run the ATS optimization tool."""
    # Initialize models and templates
    print("Initializing models...")
    
    # Initialize AI model
    try:
        # Try to use Hugging Face model
        model = HuggingFaceModel("mistralai/Mistral-7B-Instruct-v0.2")
        print("Using Hugging Face Mistral-7B model")
    except Exception as e:
        # Fall back to OpenAI if there's an issue
        print(f"Couldn't initialize Hugging Face model: {e}")
        print("Falling back to OpenAI")
        from langchain_openai.chat_models import ChatOpenAI
        model = ChatOpenAI(model="gpt-4")
    
    # Initialize SBERT model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Define prompt templates
    resume_template = SimplePromptTemplate("""
        Improve the resume bullet points to match the job description while keeping them realistic and professional.
        Provide 3 optimized bullet points.

        Job Description:
        {job_description}

        Resume Excerpt:
        {resume_text}

        Improved Bullet Points:
    """)

    cover_letter_template = SimplePromptTemplate("""
        Write a compelling, one-page cover letter tailored to the job description below.
        Ensure it includes a strong opening, highlights key skills, and ends with a confident closing.

        Job Description:
        {job_description}

        Resume Excerpt:
        {resume_text}

        Cover Letter:
    """)
    
    # Get resume from user
    print("Please select your resume file...")
    Tk().withdraw()
    resume_path = askopenfilename(title="Select Resume File (PDF, DOCX, or TXT)")
    
    if not resume_path:
        print("No file selected. Exiting.")
        return
    
    # Extract and preprocess resume text
    print("Extracting resume content...")
    resume_text = extract_text(resume_path)
    preprocessed_resume = preprocess_text(resume_text)
    
    # Process job descriptions
    folder_path = my_path
    similarity_scores = {}
    
    print("Analyzing job descriptions...")
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # Extract and preprocess job description
            job_description = extract_text_from_txt(file_path)
            preprocessed_job_description = preprocess_text(job_description)
            
            # Calculate similarities
            tfidf_score = round(calculate_tfidf_similarity(
                preprocessed_resume, preprocessed_job_description), 2)
            
            sbert_score = round(calculate_sbert_similarity(
                preprocessed_resume, preprocessed_job_description, sbert_model), 2)
            
            # Get important words and missing keywords
            resume_top_words = get_top_tfidf_words(preprocessed_resume)
            job_top_words = get_top_tfidf_words(preprocessed_job_description)
            missing_keywords = find_missing_keywords(
                preprocessed_job_description, preprocessed_resume)
            
            # Store results
            similarity_scores[filename] = {
                'tfidf_score': tfidf_score, 
                'sbert_score': sbert_score, 
                'jd_top_words': job_top_words, 
                'resume_top_words': resume_top_words, 
                'missing_keywords': missing_keywords
            }
    
    if not similarity_scores:
        print(f"No job descriptions found in {folder_path}. Exiting.")
        return
    
    # Sort by SBERT score (more accurate)
    sorted_similarity_scores = sorted(
        similarity_scores.items(), 
        key=lambda x: x[1]['sbert_score'], 
        reverse=True
    )
    
    print(f"Found {len(sorted_similarity_scores)} matching job descriptions")
    top_matches = sorted_similarity_scores[:5]  # Get top 5 matching companies
    
    # Generate improvements for top matches
    results = {}
    print("Creating AI-Suggested Resume Improvements and Cover Letters...")
    for company, score_data in top_matches:
        file_path = os.path.join(folder_path, company)
        
        # Extract job description
        job_description = extract_text_from_txt(file_path)
        
        # Generate improvements
        print(f"Processing {company}...")
        improved_bullets = improve_resume_bullet_points(
            resume_text, job_description, model, resume_template)
        cover_letter = generate_cover_letter(
            job_description, resume_text, model, cover_letter_template)
        
        # Store results
        results[company] = {
            "score": score_data['sbert_score'],
            "missing_keywords": score_data['missing_keywords'],
            "improved_bullets": improved_bullets,
            "cover_letter": cover_letter
        }
    
    # Display results
    for company, data in results.items():
        sbert_score = data['score']
        missing_keywords = data['missing_keywords']
        improved_bullets = data['improved_bullets']
        cover_letter = data['cover_letter']
        
        print("\n" + "="*50)
        print(f"JOB: {company}")
        print(f"MATCH SCORE: {sbert_score}%")
        
        print("\n MISSING KEYWORDS:")
        if missing_keywords:
            print(", ".join(missing_keywords))
        else:
            print("No critical keywords missing!")
        
        print("\nAI-SUGGESTED RESUME IMPROVEMENTS:")
        for bullet in improved_bullets:
            print(f"- {bullet}")
        
        print("\nAI-GENERATED COVER LETTER:")
        print(cover_letter)
        print("="*50)


if __name__ == "__main__":
    main()
