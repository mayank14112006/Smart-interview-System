from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from textblob import TextBlob
from rake_nltk import Rake
import csv
from pathlib import Path

import nltk
import os

nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))


nltk.download('punkt_tab')

app = FastAPI()

# Allow frontend access (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keyword setup from CSV
keyword_map = {}

keywords_path = "C:\\Users\\MAYANK GOEL\\smart interview\\backend\\db\\keywords.csv"
if Path(keywords_path).exists():
    with open(keywords_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = row["question"].strip()
            keywords = [k.strip().lower() for k in row["keywords"].split(",")]
            keyword_map[question] = keywords
else:
    print("⚠️ keywords.csv not found!")

rake = Rake()

# Request/Response Models
class ResponseItem(BaseModel):
    question: str
    answer: str

class InterviewData(BaseModel):
    responses: List[ResponseItem]

# POST endpoint
@app.post("/process")
async def process_answers(data: InterviewData):
    feedback = []
    total_sentiment = 0

    ffeedback = []
    sum_scores = 0

    for r in data.responses:
        ans = r.answer or ""
        polarity = TextBlob(ans).sentiment.polarity

        # SENTIMENT
        sentiment_sub = (polarity + 1) / 2 * 50

        # KEYWORDS
        rake.extract_keywords_from_text(ans)
        extracted = rake.get_ranked_phrases()
        expected = keyword_map.get(r.question.strip(), [])
        matched = [kw for kw in expected if kw in ans.lower()]
        keyword_match = (len(matched) / len(expected) * 100) if expected else 0
        keyword_sub = keyword_match / 100 * 30

        # RELEVANCE (using same polarity mapping, or sentiment-based)
        relevance_score = round(polarity * 100)
        relevance_sub = max(0, min(100, relevance_score)) / 100 * 20

        # Total per-question
        question_score = sentiment_sub + keyword_sub + relevance_sub
        sum_scores += question_score

        feedback.append({
            "question": r.question,
            "answer": ans,
            "sentiment": polarity,
            "keyword_match_score": round(keyword_match),
            "relevance_score": relevance_score,
            "question_score": round(question_score),
            "matched_keywords": matched
        })

    # Average into 0–100
    overall = round(sum_scores / len(data.responses))
    overall = max(0, min(100, overall))

    return {
        "overall_score": overall,
        "per_question_feedback": feedback
    }
