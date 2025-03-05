# 🚀 AI-Powered Resume Optimizer & Job Matching Tool

## Overview

This advanced AI-powered tool helps job seekers optimize their resumes, find the most relevant job opportunities, and generate tailored cover letters. By leveraging cutting-edge natural language processing and machine learning techniques, the Resume Optimizer provides intelligent insights to improve your job application strategy.

## 🌟 Key Features

- **Intelligent Job Matching**: Advanced similarity scoring using TF-IDF and Sentence-BERT embeddings
- **Resume Enhancement**: AI-powered suggestions for improving resume bullet points
- **Automated Cover Letter Generation**: Create personalized cover letters for each job application
- **Multi-Format Resume Support**: Works with PDF, DOCX, and TXT resume formats
- **Keyword Analysis**: Identifies missing keywords in your resume compared to job descriptions

## 🛠 Tech Stack

- Python 3.8+
- Natural Language Processing Libraries:
  - NLTK
  - Sentence Transformers
  - Scikit-learn
- Machine Learning Models:
  - Hugging Face Mistral-7B
  - Sentence-BERT
- Web Scraping:
  - Selenium
  - BeautifulSoup
- Other Key Libraries:
  - PyPDF2
  - docx2txt
  - Pandas

## 📋 Prerequisites

- Python 3.8 or higher
- Hugging Face Account (for model access)
- Selenium WebDriver
- Stable internet connection

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-optimizer.git
cd resume-optimizer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
- Create a `config.py` file with your `my_path` for job descriptions
- Set up Hugging Face access token in the script

## 🚀 Usage

1. **Job Scraping**:
```bash
python linkedin_job_scraper.py
# Enter job title and location when prompted
```

2. **Resume Optimization**:
```bash
python new-cleaned-ats-script.py
# Select your resume file when prompted
```

## 📊 How It Works

1. **Job Discovery**
   - Scrape LinkedIn jobs based on user-specified job title and location
   - Save job descriptions as individual text files

2. **Resume Analysis**
   - Extract and preprocess resume text
   - Calculate similarity scores between resume and job descriptions
   - Identify missing keywords

3. **AI Enhancements**
   - Generate optimized resume bullet points
   - Create personalized cover letters
   - Provide detailed job matching insights

## 🔍 Similarity Scoring Methods

- **TF-IDF Similarity**: Traditional keyword-based matching
- **Sentence-BERT Similarity**: Advanced semantic understanding
- Scores mapped to 50-100% range for easy interpretation

## 🛡 Limitations

- Requires an active internet connection
- Job matching accuracy depends on job description quality
- Cover letter generation is AI-assisted and may need human review

## 🤝 Contributing

Contributions are welcome! Please:
- Fork the repository
- Create a feature branch
- Submit a pull request

## 📄 License

Nothing yet...

## 🙌 Acknowledgments

- Hugging Face for transformers
- LinkedIn for job data
- Open-source community

## 📧 Contact

Name : Jake Byford

Email : jake_byford@outlook.com

Website : jakebyford.com

