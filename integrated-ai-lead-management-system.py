# Integrated AI Lead Management System
# Combines an AI-powered chatbot for lead automation with an AI-driven CRM

# Required packages:
# pip install fastapi uvicorn transformers python-dotenv sqlalchemy pymongo pandas scikit-learn nltk requests python-whatsapp-api

import os
import json
import logging
import datetime
import random
from typing import List, Dict, Optional, Any
from enum import Enum
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from datetime import timedelta
import asyncio
import dotenv
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load environment variables
dotenv.load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Lead Management System", description="Integrated AI chatbot and CRM system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./lead_management.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# NLP Setup
nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()

# Load ML models
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
lead_scoring_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
intent_recognition = pipeline("text-classification", model="distilbert-base-uncased")

# Microsoft Dynamics 365 integration
DYNAMICS_API_URL = os.getenv("DYNAMICS_API_URL")
DYNAMICS_API_KEY = os.getenv("DYNAMICS_API_KEY")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger("lead_management")

# ------------------ Database Models ------------------

class Source(Enum):
    WEBSITE = "website"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    CHATBOT = "chatbot"
    REFERRAL = "referral"
    OTHER = "other"

class LeadStatus(Enum):
    NEW = "new"
    QUALIFIED = "qualified"
    CONTACTED = "contacted"
    NURTURING = "nurturing"
    OPPORTUNITY = "opportunity"
    CUSTOMER = "customer"
    LOST = "lost"

class Lead(Base):
    __tablename__ = "leads"
    
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, index=True)
    last_name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    phone = Column(String)
    company = Column(String)
    job_title = Column(String)
    industry = Column(String)
    source = Column(String)
    status = Column(String, default=LeadStatus.NEW.value)
    score = Column(Float, default=0.0)
    budget_range = Column(String)
    timeline = Column(String)
    requirements = Column(Text)
    notes = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_contact = Column(DateTime, nullable=True)
    language = Column(String, default="en")
    
    interactions = relationship("Interaction", back_populates="lead")
    tasks = relationship("Task", back_populates="lead")

class Interaction(Base):
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))
    channel = Column(String)  # email, phone, chatbot, social_media
    direction = Column(String)  # inbound, outbound
    content = Column(Text)
    sentiment_score = Column(Float)
    created_at = Column(DateTime, default=func.now())
    
    lead = relationship("Lead", back_populates="interactions")

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    description = Column(Text)
    due_date = Column(DateTime)
    status = Column(String)  # pending, completed, overdue
    priority = Column(String)  # low, medium, high
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    
    lead = relationship("Lead", back_populates="tasks")
    user = relationship("User", back_populates="tasks")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    role = Column(String)  # admin, sales_rep, manager
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    
    tasks = relationship("Task", back_populates="user")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), nullable=True)
    visitor_ip = Column(String)
    language = Column(String, default="en")
    started_at = Column(DateTime, default=func.now())
    ended_at = Column(DateTime, nullable=True)
    qualified = Column(Boolean, default=False)
    
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    sender = Column(String)  # "bot" or "user"
    message = Column(Text)
    intent = Column(String, nullable=True)
    sentiment = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=func.now())
    
    session = relationship("ChatSession", back_populates="messages")

# Create all tables
Base.metadata.create_all(bind=engine)

# ------------------ Pydantic Models ------------------

class LeadBase(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    phone: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    industry: Optional[str] = None
    source: str
    budget_range: Optional[str] = None
    timeline: Optional[str] = None
    requirements: Optional[str] = None
    notes: Optional[str] = None
    language: str = "en"

class LeadCreate(LeadBase):
    pass

class LeadUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    industry: Optional[str] = None
    status: Optional[str] = None
    score: Optional[float] = None
    budget_range: Optional[str] = None
    timeline: Optional[str] = None
    requirements: Optional[str] = None
    notes: Optional[str] = None
    language: Optional[str] = None

class LeadResponse(LeadBase):
    id: int
    status: str
    score: float
    created_at: datetime.datetime
    updated_at: datetime.datetime
    last_contact: Optional[datetime.datetime] = None

    class Config:
        orm_mode = True

class InteractionCreate(BaseModel):
    lead_id: int
    channel: str
    direction: str
    content: str

class TaskCreate(BaseModel):
    lead_id: int
    user_id: int
    title: str
    description: Optional[str] = None
    due_date: datetime.datetime
    priority: str = "medium"

class ChatMessageCreate(BaseModel):
    message: str
    sender: str = "user"
    session_id: str

class ChatResponse(BaseModel):
    message: str
    session_id: str
    lead_created: bool = False
    lead_id: Optional[int] = None

class AnalyticsResponse(BaseModel):
    total_leads: int
    leads_by_source: Dict[str, int]
    leads_by_status: Dict[str, int]
    conversion_rate: float
    average_score: float
    interactions_count: int

# ------------------ Dependency ------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------------ Chatbot Logic ------------------

# Define conversation flow
conversation_flow = {
    "greeting": {
        "responses": [
            "Hello! Welcome to our website. How can I help you today?",
            "Hi there! I'm here to assist you. What brings you to our website today?",
            "Welcome! I'm your virtual assistant. What are you looking for today?"
        ],
        "next_states": ["identify_need", "collect_info"]
    },
    "identify_need": {
        "responses": [
            "Can you tell me more about what you're looking for?",
            "What specific services or products are you interested in?",
            "What challenges are you trying to solve?"
        ],
        "next_states": ["collect_info", "provide_info"]
    },
    "collect_info": {
        "responses": [
            "Great! To better assist you, could you share your company name and industry?",
            "To help you better, I'd like to know a bit about your business. What industry are you in?",
            "Thanks for sharing. What's your company name and what industry do you operate in?"
        ],
        "next_states": ["budget_inquiry", "timeline_inquiry"]
    },
    "budget_inquiry": {
        "responses": [
            "What budget range are you considering for this project?",
            "Do you have a specific budget in mind for this solution?",
            "To recommend the right solution, could you share your approximate budget range?"
        ],
        "next_states": ["timeline_inquiry", "qualification"]
    },
    "timeline_inquiry": {
        "responses": [
            "What's your timeline for implementing this solution?",
            "When are you looking to get started with this project?",
            "Do you have a specific deadline or timeline in mind?"
        ],
        "next_states": ["qualification", "provide_info"]
    },
    "provide_info": {
        "responses": [
            "Based on what you've shared, I think we have solutions that might be a good fit. Would you like to schedule a call with one of our specialists?",
            "Thanks for the information. We have several options that could work for your needs. Would you like me to have a sales representative contact you?",
            "I appreciate you sharing those details. We have expertise in this area. Would you like to provide your email for a follow-up?"
        ],
        "next_states": ["collect_contact", "end_conversation"]
    },
    "qualification": {
        "responses": [
            "Thank you for sharing that information. Based on what you've told me, I think we can help. Would you be interested in speaking with one of our consultants?",
            "Thanks for the details. Your project seems to align well with our services. Would you like to schedule a call with our team?",
            "This sounds like a great fit for our expertise. Would you like me to arrange for someone to contact you with more information?"
        ],
        "next_states": ["collect_contact", "end_conversation"]
    },
    "collect_contact": {
        "responses": [
            "Great! Could you please provide your name and email address so we can get in touch?",
            "Excellent! If you could share your name, email, and phone number, we'll have someone reach out soon.",
            "Perfect! To set this up, I'll need your name and best contact information."
        ],
        "next_states": ["end_conversation"]
    },
    "end_conversation": {
        "responses": [
            "Thank you for your interest! A team member will contact you shortly. Is there anything else I can help you with today?",
            "Thanks for chatting with me today! Someone from our team will be in touch soon. Have a great day!",
            "I've recorded your information and a consultant will reach out soon. Thank you for your time today!"
        ],
        "next_states": []
    }
}

# Languages supported
supported_languages = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic"
}

# Translation services mock (in a real system, you'd use an API)
def translate_text(text, source_lang="en", target_lang="en"):
    if source_lang == target_lang:
        return text
    
    # Mock translation - in a real system you'd call an actual translation API
    logger.info(f"Translating from {source_lang} to {target_lang}: {text}")
    
    # In a real implementation, you would call:
    # response = requests.post(
    #     "https://api.cognitive.microsofttranslator.com/translate",
    #     headers={
    #         "Ocp-Apim-Subscription-Key": os.getenv("TRANSLATOR_API_KEY"),
    #         "Content-type": "application/json"
    #     },
    #     json=[{"text": text}],
    #     params={
    #         "api-version": "3.0",
    #         "from": source_lang,
    #         "to": target_lang
    #     }
    # )
    # return response.json()[0]["translations"][0]["text"]
    
    return f"[Translated from {source_lang} to {target_lang}]: {text}"

# Intent detection
def detect_intent(message):
    # In a real system, you'd use a properly trained model
    # This is a simplified mock implementation
    message = message.lower()
    
    intents = {
        "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"],
        "information": ["information", "details", "tell me", "know more", "learn about"],
        "pricing": ["price", "cost", "budget", "pricing", "expensive", "cheap"],
        "features": ["features", "capabilities", "what can", "functions", "options"],
        "contact": ["contact", "email", "call", "phone", "reach"],
        "complaint": ["problem", "issue", "not working", "broken", "disappointed", "unhappy"],
        "gratitude": ["thank", "thanks", "appreciate", "grateful"],
        "goodbye": ["bye", "goodbye", "see you", "talk later", "end"]
    }
    
    detected_intent = "unknown"
    highest_score = 0
    
    for intent, keywords in intents.items():
        score = sum(1 for keyword in keywords if keyword in message)
        if score > highest_score:
            highest_score = score
            detected_intent = intent
    
    return detected_intent

# Entity extraction
def extract_entities(message):
    # Simplified entity extraction - in a real system, use NER models
    entities = {}
    
    # Email extraction
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    import re
    email_match = re.search(email_pattern, message)
    if email_match:
        entities["email"] = email_match.group(0)
    
    # Phone extraction
    phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phone_match = re.search(phone_pattern, message)
    if phone_match:
        entities["phone"] = phone_match.group(0)
    
    # Company extraction (simplified)
    company_indicators = ["at", "for", "with", "company", "organization", "firm"]
    words = message.split()
    for i, word in enumerate(words):
        if word.lower() in company_indicators and i + 1 < len(words):
            if words[i+1][0].isupper():  # Check if next word starts with capital letter
                entities["company"] = words[i+1].strip(".,;:")
    
    # Budget extraction
    budget_pattern = r'\$\s?(\d+(?:[.,]\d+)?(?:\s?[kmbt])?)'
    budget_match = re.search(budget_pattern, message, re.IGNORECASE)
    if budget_match:
        entities["budget"] = budget_match.group(0)
    
    return entities

# Sentiment analysis
def analyze_sentiment(message):
    sentiment = sentiment_analyzer.polarity_scores(message)
    return sentiment["compound"]  # Returns a value from -1 (negative) to 1 (positive)

# Lead qualification scoring
def score_lead(lead_data):
    # This would be a trained ML model in production
    # Here it's a simplified rule-based approach
    
    score = 50  # Base score
    
    # Industry scoring
    high_value_industries = ["technology", "healthcare", "finance", "manufacturing", "education"]
    if lead_data.get("industry", "").lower() in high_value_industries:
        score += 15
    
    # Budget scoring
    budget = lead_data.get("budget_range", "")
    if "enterprise" in budget.lower() or "unlimited" in budget.lower():
        score += 20
    elif "mid" in budget.lower():
        score += 10
    
    # Timeline scoring
    timeline = lead_data.get("timeline", "")
    if "immediate" in timeline.lower() or "urgent" in timeline.lower() or "asap" in timeline.lower():
        score += 15
    elif "month" in timeline.lower():
        score += 10
    
    # Company size and completeness of information can also factor in
    if lead_data.get("company"):
        score += 5
    if lead_data.get("job_title"):
        score += 5
    
    # Normalize to 0-100
    return min(max(score, 0), 100)

# Process chatbot message and get response
async def process_chatbot_message(message, session_id, db: Session):
    # Get or create chat session
    chat_session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not chat_session:
        chat_session = ChatSession(
            session_id=session_id,
            visitor_ip="127.0.0.1",  # In real app, get from request
            language="en"
        )
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
        current_state = "greeting"
    else:
        # Determine current state based on conversation history
        last_bot_message = db.query(ChatMessage).filter(
            ChatMessage.session_id == chat_session.id,
            ChatMessage.sender == "bot"
        ).order_by(ChatMessage.timestamp.desc()).first()
        
        if last_bot_message:
            # This is simplified - in a real system, you'd use the bot's last intent
            for state, data in conversation_flow.items():
                if any(response in last_bot_message.message for response in data["responses"]):
                    current_state = state
                    break
            else:
                current_state = "identify_need"
        else:
            current_state = "greeting"
    
    # Process user message
    intent = detect_intent(message)
    entities = extract_entities(message)
    sentiment = analyze_sentiment(message)
    
    # Save user message
    user_message = ChatMessage(
        session_id=chat_session.id,
        sender="user",
        message=message,
        intent=intent,
        sentiment=sentiment
    )
    db.add(user_message)
    
    # Update lead information if entities were found
    lead_created = False
    lead_id = None
    
    if chat_session.lead_id:
        # Update existing lead
        lead = db.query(Lead).filter(Lead.id == chat_session.lead_id).first()
        
        if "email" in entities and not lead.email:
            lead.email = entities["email"]
        if "phone" in entities and not lead.phone:
            lead.phone = entities["phone"]
        if "company" in entities and not lead.company:
            lead.company = entities["company"]
        if "budget" in entities and not lead.budget_range:
            lead.budget_range = entities["budget"]
        
        db.commit()
        lead_id = lead.id
    elif "email" in entities:
        # Check if lead with this email already exists
        existing_lead = db.query(Lead).filter(Lead.email == entities["email"]).first()
        
        if existing_lead:
            chat_session.lead_id = existing_lead.id
            lead_id = existing_lead.id
        else:
            # Create new lead
            new_lead = Lead(
                first_name="",  # To be updated later
                last_name="",   # To be updated later
                email=entities["email"],
                phone=entities.get("phone", ""),
                company=entities.get("company", ""),
                source="chatbot",
                budget_range=entities.get("budget", ""),
                language=chat_session.language
            )
            db.add(new_lead)
            db.commit()
            db.refresh(new_lead)
            
            chat_session.lead_id = new_lead.id
            lead_id = new_lead.id
            lead_created = True
    
    # Determine next state and response
    if intent == "goodbye":
        next_state = "end_conversation"
    elif chat_session.qualified:
        next_state = "collect_contact" if not lead_id else "end_conversation"
    else:
        possible_next_states = conversation_flow[current_state]["next_states"]
        next_state = random.choice(possible_next_states) if possible_next_states else "end_conversation"
    
    # If the state is qualification, evaluate whether the lead is qualified
    if next_state == "qualification" and chat_session.lead_id:
        lead = db.query(Lead).filter(Lead.id == chat_session.lead_id).first()
        lead_data = {
            "industry": lead.industry,
            "budget_range": lead.budget_range,
            "timeline": lead.timeline,
            "company": lead.company,
            "job_title": lead.job_title
        }
        score = score_lead(lead_data)
        lead.score = score
        chat_session.qualified = score >= 60  # Set qualification threshold
        db.commit()
    
    # Get bot response
    if conversation_flow.get(next_state):
        bot_response = random.choice(conversation_flow[next_state]["responses"])
    else:
        bot_response = "I'm not sure how to respond to that. Could you please rephrase or provide more information?"
    
    # Save bot message
    bot_message = ChatMessage(
        session_id=chat_session.id,
        sender="bot",
        message=bot_response,
        intent=next_state,
        sentiment=0.0  # Neutral sentiment for bot messages
    )
    db.add(bot_message)
    db.commit()
    
    # Update chat session with lead info if created
    if lead_created and chat_session.lead_id:
        chat_session.lead_id = lead_id
        db.commit()
    
    return ChatResponse(
        message=bot_response,
        session_id=session_id,
        lead_created=lead_created,
        lead_id=lead_id
    )

# ------------------ CRM Logic ------------------

# Lead scoring model
def calculate_lead_score(lead, interactions, db: Session):
    # In a real system, this would use a trained ML model
    # Here we'll use a simplified rule-based approach
    
    # Base score: 0-100
    score = 50
    
    # Interaction recency and frequency
    if interactions:
        # More recent interactions boost score
        last_interaction = max(interaction.created_at for interaction in interactions)
        days_since_last = (datetime.datetime.now() - last_interaction).days
        recency_score = max(0, 10 - days_since_last)
        score += recency_score
        
        # More interactions boost score
        freq_score = min(10, len(interactions))
        score += freq_score
        
        # Positive sentiment interactions boost score
        avg_sentiment = sum(interaction.sentiment_score for interaction in interactions) / len(interactions)
        sentiment_score = int(avg_sentiment * 10)
        score += sentiment_score
    
    # Industry factor
    high_value_industries = ["technology", "healthcare", "finance", "insurance", "manufacturing"]
    if lead.industry and lead.industry.lower() in high_value_industries:
        score += 10
    
    # Budget factor
    if lead.budget_range:
        budget = lead.budget_range.lower()
        if "enterprise" in budget or "high" in budget or "unlimited" in budget:
            score += 15
        elif "medium" in budget or "mid" in budget:
            score += 10
    
    # Timeline factor
    if lead.timeline:
        timeline = lead.timeline.lower()
        if "immediate" in timeline or "urgent" in timeline or "asap" in timeline:
            score += 15
        elif "month" in timeline or "soon" in timeline:
            score += 10
    
    # Cap the score between 0-100
    return max(0, min(100, score))

# Lead prioritization
def get_prioritized_leads(db: Session, user_id: int, limit: int = 10):
    # Get all leads assigned to the user
    leads = db.query(Lead).all()
    
    # For each lead, calculate current score
    prioritized_leads = []
    for lead in leads:
        interactions = db.query(Interaction).filter(Interaction.lead_id == lead.id).all()
        current_score = calculate_lead_score(lead, interactions, db)
        
        # Update score in database
        lead.score = current_score
        db.commit()
        
        prioritized_leads.append({
            "lead": lead,
            "score": current_score,
            "last_contact": lead.last_contact or datetime.datetime.min,
            "days_since_contact": (datetime.datetime.now() - (lead.last_contact or datetime.datetime.min)).days
        })
    
    # Sort by score (descending) and days since last contact (descending)
    prioritized_leads.sort(key=lambda x: (x["score"], x["days_since_contact"]), reverse=True)
    
    # Return top N leads
    return [item["lead"] for item in prioritized_leads[:limit]]

# Automated follow-up scheduling
def schedule_followup(lead_id: int, user_id: int, db: Session):
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        return False
    
    # Determine appropriate follow-up type based on lead status and score
    if lead.status == LeadStatus.NEW.value:
        task_title = "Initial contact with new lead"
        due_date = datetime.datetime.now() + datetime.timedelta(days=1)
        priority = "high"
    elif lead.status == LeadStatus.CONTACTED.value:
        task_title = "Follow-up call after initial contact"
        due_date = datetime.datetime.now() + datetime.timedelta(days=3)
        priority = "medium"
    elif lead.status == LeadStatus.NURTURING.value:
        task_title = "Send additional information"
        due_date = datetime.datetime.now() + datetime.timedelta(days=7)
        priority = "medium"
    elif lead.status == LeadStatus.OPPORTUNITY.value:
        task_title = "Prepare and send proposal"
        due_date = datetime.datetime.now() + datetime.timedelta(days=2)
        priority = "high"
    else:
        task_title = "Follow up with lead"
        due_date = datetime.datetime.now() + datetime.timedelta(days=5)
        priority = "medium"
    
    # Create follow-up task
    new_task = Task(
        lead_id=lead_id,
        user_id=user_id,
        title=task_title,
        description=f"Follow up with {lead.first_name} {lead.last_name} from {lead.company or 'unknown company'}",
        due_date=due_date,
        status="pending",
        priority=priority
    )
    
    db.add(new_task)
    db.commit()
    
    return True

# CRM analytics
def generate_analytics(db: Session):
    # Total leads
    total_leads = db.query(Lead).count()
    
    # Leads by source
    leads_by_source = {}
    sources = db.query(Lead.source, func.count(Lead.id)).group_by(Lead.source).all()
    for source, count in sources:
        leads_by_source[source] = count
    
    # Leads by status
    leads_by_status = {}
    statuses = db.query(Lead.status, func.count(Lead.id)).group_by(Lead.status).all()
    for status, count in statuses:
        leads_by_status[status] = count
    
    # Conversion rate (leads that became customers)
    customer_leads = db.query(Lead).filter(Lead.status == LeadStatus.CUSTOMER.value).count()
    conversion_rate = (customer_leads / total_leads) * 100 if total_leads > 0 else 0
    
    # Average lead score
    avg_score = db.query(func.avg(Lead.score)).scalar() or 0
    
    # Interaction count
    interactions_count = db.query(Interaction).count()
    
    return AnalyticsResponse(
        total_leads=total_leads,
        leads_by_source=leads_by_source,
        leads_by_status=leads_by_status,
        conversion_rate=conversion_rate,
        average_score=float(avg_score),
        interactions_count=interactions_count
    )
# ------------------ API Endpoints ------------------

# Chatbot API endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    message: ChatMessageCreate,
    db: Session = Depends(get_db)
):
    # Process the chat message and get a response
    response = await process_chatbot_message(message.message, message.session_id, db)
    return response

@app.websocket("/ws/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, db: Session = Depends(get_db)):
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            # Process message
            response = await process_chatbot_message(data, session_id, db)
            
            # Send response back to client
            await websocket.send_json(response.dict())
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")

# Lead API endpoints
@app.post("/api/leads", response_model=LeadResponse)
def create_lead(lead: LeadCreate, db: Session = Depends(get_db)):
    # Check if lead with this email already exists
    existing_lead = db.query(Lead).filter(Lead.email == lead.email).first()
    if existing_lead:
        raise HTTPException(status_code=400, detail="Lead with this email already exists")
    
    # Create new lead
    db_lead = Lead(
        first_name=lead.first_name,
        last_name=lead.last_name,
        email=lead.email,
        phone=lead.phone,
        company=lead.company,
        job_title=lead.job_title,
        industry=lead.industry,
        source=lead.source,
        budget_range=lead.budget_range,
        timeline=lead.timeline,
        requirements=lead.requirements,
        notes=lead.notes,
        language=lead.language,
        status=LeadStatus.NEW.value
    )
    
    db.add(db_lead)
    db.commit()
    db.refresh(db_lead)
    
    # Calculate initial score
    db_lead.score = score_lead({
        "industry": db_lead.industry,
        "budget_range": db_lead.budget_range,
        "timeline": db_lead.timeline,
        "company": db_lead.company,
        "job_title": db_lead.job_title
    })
    db.commit()
    
    # Schedule initial follow-up if source is not chatbot (chatbot has its own flow)
    if db_lead.source != "chatbot":
        schedule_followup(db_lead.id, 1, db)  # Assuming user_id 1 is the default user
    
    # Integrate with Microsoft Dynamics 365
    try:
        sync_lead_with_dynamics(db_lead)
    except Exception as e:
        logger.error(f"Failed to sync lead with Dynamics 365: {str(e)}")
    
    return db_lead

@app.get("/api/leads", response_model=List[LeadResponse])
def get_leads(
    skip: int = 0, 
    limit: int = 100, 
    status: Optional[str] = None,
    source: Optional[str] = None,
    min_score: Optional[float] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Lead)
    
    # Apply filters
    if status:
        query = query.filter(Lead.status == status)
    if source:
        query = query.filter(Lead.source == source)
    if min_score is not None:
        query = query.filter(Lead.score >= min_score)
    
    # Pagination
    leads = query.offset(skip).limit(limit).all()
    
    return leads

@app.get("/api/leads/{lead_id}", response_model=LeadResponse)
def get_lead(lead_id: int, db: Session = Depends(get_db)):
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    return lead

@app.put("/api/leads/{lead_id}", response_model=LeadResponse)
def update_lead(lead_id: int, lead_update: LeadUpdate, db: Session = Depends(get_db)):
    db_lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if db_lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    # Update lead attributes that are provided
    update_data = lead_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_lead, key, value)
    
    # Recalculate score if relevant fields changed
    if any(field in update_data for field in ["industry", "budget_range", "timeline"]):
        db_lead.score = score_lead({
            "industry": db_lead.industry,
            "budget_range": db_lead.budget_range,
            "timeline": db_lead.timeline,
            "company": db_lead.company,
            "job_title": db_lead.job_title
        })
    
    db.commit()
    db.refresh(db_lead)
    
    # Sync with Microsoft Dynamics 365
    try:
        sync_lead_with_dynamics(db_lead)
    except Exception as e:
        logger.error(f"Failed to sync lead with Dynamics 365: {str(e)}")
    
    return db_lead

@app.delete("/api/leads/{lead_id}")
def delete_lead(lead_id: int, db: Session = Depends(get_db)):
    db_lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if db_lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    # Delete all related interactions and tasks
    db.query(Interaction).filter(Interaction.lead_id == lead_id).delete()
    db.query(Task).filter(Task.lead_id == lead_id).delete()
    
    # Delete lead
    db.delete(db_lead)
    db.commit()
    
    return {"message": "Lead deleted successfully"}

# Interaction API endpoints
@app.post("/api/interactions")
def create_interaction(interaction: InteractionCreate, db: Session = Depends(get_db)):
    # Verify lead exists
    lead = db.query(Lead).filter(Lead.id == interaction.lead_id).first()
    if lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    # Analyze sentiment
    sentiment_score = analyze_sentiment(interaction.content)
    
    # Create interaction
    db_interaction = Interaction(
        lead_id=interaction.lead_id,
        channel=interaction.channel,
        direction=interaction.direction,
        content=interaction.content,
        sentiment_score=sentiment_score
    )
    
    db.add(db_interaction)
    
    # Update lead's last_contact date
    lead.last_contact = datetime.datetime.now()
    
    # Update lead status based on interaction
    if lead.status == LeadStatus.NEW.value and interaction.direction == "outbound":
        lead.status = LeadStatus.CONTACTED.value
    
    db.commit()
    db.refresh(db_interaction)
    
    # Schedule follow-up based on interaction
    schedule_followup(interaction.lead_id, 1, db)  # Assuming user_id 1 is the default user
    
    return db_interaction

@app.get("/api/leads/{lead_id}/interactions")
def get_lead_interactions(lead_id: int, db: Session = Depends(get_db)):
    # Verify lead exists
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    # Get interactions
    interactions = db.query(Interaction).filter(Interaction.lead_id == lead_id).all()
    
    return interactions

# Task API endpoints
@app.post("/api/tasks")
def create_task(task: TaskCreate, db: Session = Depends(get_db)):
    # Verify lead exists
    lead = db.query(Lead).filter(Lead.id == task.lead_id).first()
    if lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    # Create task
    db_task = Task(
        lead_id=task.lead_id,
        user_id=task.user_id,
        title=task.title,
        description=task.description,
        due_date=task.due_date,
        status="pending",
        priority=task.priority
    )
    
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    
    return db_task

@app.put("/api/tasks/{task_id}/complete")
def complete_task(task_id: int, db: Session = Depends(get_db)):
    # Verify task exists
    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update task status
    task.status = "completed"
    task.completed_at = datetime.datetime.now()
    
    db.commit()
    
    return {"message": "Task completed successfully"}

@app.get("/api/users/{user_id}/tasks")
def get_user_tasks(
    user_id: int, 
    status: Optional[str] = None,
    priority: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Task).filter(Task.user_id == user_id)
    
    # Apply filters
    if status:
        query = query.filter(Task.status == status)
    if priority:
        query = query.filter(Task.priority == priority)
    
    # Get tasks
    tasks = query.all()
    
    return tasks

# Dashboard and analytics endpoints
@app.get("/api/analytics", response_model=AnalyticsResponse)
def get_analytics(db: Session = Depends(get_db)):
    return generate_analytics(db)

@app.get("/api/dashboard/prioritized-leads")
def get_prioritized_leads_endpoint(user_id: int = 1, limit: int = 10, db: Session = Depends(get_db)):
    leads = get_prioritized_leads(db, user_id, limit)
    return leads

@app.get("/api/dashboard/upcoming-tasks")
def get_upcoming_tasks(user_id: int = 1, days: int = 7, db: Session = Depends(get_db)):
    end_date = datetime.datetime.now() + datetime.timedelta(days=days)
    
    tasks = db.query(Task).filter(
        Task.user_id == user_id,
        Task.status == "pending",
        Task.due_date <= end_date
    ).order_by(Task.due_date).all()
    
    return tasks

@app.get("/api/dashboard/conversion-stats")
def get_conversion_stats(period: str = "month", db: Session = Depends(get_db)):
    today = datetime.datetime.now()
    
    if period == "week":
        start_date = today - datetime.timedelta(days=7)
    elif period == "month":
        start_date = today - datetime.timedelta(days=30)
    elif period == "quarter":
        start_date = today - datetime.timedelta(days=90)
    else:  # default to year
        start_date = today - datetime.timedelta(days=365)
    
    # Get leads created in the period
    new_leads = db.query(Lead).filter(Lead.created_at >= start_date).count()
    
    # Get conversions in the period
    conversions = db.query(Lead).filter(
        Lead.status == LeadStatus.CUSTOMER.value,
        Lead.updated_at >= start_date
    ).count()
    
    # Calculate conversion rate
    conversion_rate = (conversions / new_leads) * 100 if new_leads > 0 else 0
    
    return {
        "period": period,
        "new_leads": new_leads,
        "conversions": conversions,
        "conversion_rate": conversion_rate
    }

# ------------------ Integration Functions ------------------

# Microsoft Dynamics 365 integration
def sync_lead_with_dynamics(lead):
    """
    Sync lead data with Microsoft Dynamics 365 CRM.
    In a real implementation, this would call the Dynamics 365 API.
    """
    logger.info(f"Syncing lead {lead.id} with Dynamics 365")
    
    # In a real implementation, you would do something like:
    # dynamics_data = {
    #     "firstname": lead.first_name,
    #     "lastname": lead.last_name,
    #     "emailaddress1": lead.email,
    #     "telephone1": lead.phone,
    #     "companyname": lead.company,
    #     "jobtitle": lead.job_title,
    #     "industrycode": lead.industry,
    #     "leadsourcecode": lead.source,
    #     "description": lead.requirements
    # }
    # 
    # response = requests.post(
    #     f"{DYNAMICS_API_URL}/leads",
    #     headers={"Authorization": f"Bearer {DYNAMICS_API_KEY}"},
    #     json=dynamics_data
    # )
    # 
    # if response.status_code == 201:
    #     # Store the Dynamics lead ID for future sync
    #     dynamics_id = response.json().get("leadid")
    #     lead.dynamics_id = dynamics_id
    #     db.commit()
    
    # For now, just log that we would sync
    return {"status": "success", "message": "Lead synced with Dynamics 365"}

# Email automation for follow-ups
def send_automated_email(lead_id, template_name):
    """
    Send an automated email to a lead based on a template.
    In a real implementation, this would use an email service like SendGrid or SMTP.
    """
    # In a real implementation, you would:
    # 1. Get the lead data
    # 2. Get the email template
    # 3. Substitute variables in the template
    # 4. Send the email
    
    logger.info(f"Sending {template_name} email to lead {lead_id}")
    return {"status": "success", "message": f"Email {template_name} sent to lead {lead_id}"}

# WhatsApp integration for messaging
def send_whatsapp_message(lead_id, template_name):
    """
    Send a WhatsApp message to a lead.
    In a real implementation, this would use the WhatsApp Business API.
    """
    logger.info(f"Sending {template_name} WhatsApp message to lead {lead_id}")
    return {"status": "success", "message": f"WhatsApp message {template_name} sent to lead {lead_id}"}

# ------------------ Background Tasks ------------------

# Lead scoring background task
async def update_all_lead_scores(db: Session):
    """
    Update scores for all leads periodically.
    """
    leads = db.query(Lead).all()
    for lead in leads:
        interactions = db.query(Interaction).filter(Interaction.lead_id == lead.id).all()
        lead.score = calculate_lead_score(lead, interactions, db)
    
    db.commit()
    logger.info(f"Updated scores for {len(leads)} leads")

# Task reminder background task
async def send_task_reminders(db: Session):
    """
    Send reminders for tasks due soon.
    """
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    tasks = db.query(Task).filter(
        Task.status == "pending",
        Task.due_date <= tomorrow
    ).all()
    
    for task in tasks:
        # In a real implementation, send an email or notification to the user
        logger.info(f"Sending reminder for task {task.id} to user {task.user_id}")
    
    logger.info(f"Sent reminders for {len(tasks)} tasks")

# Lead follow-up background task
async def process_lead_followups(db: Session):
    """
    Check for leads that need follow-up and create tasks or send automated messages.
    """
    # Find leads that haven't been contacted in 3+ days
    three_days_ago = datetime.datetime.now() - datetime.timedelta(days=3)
    leads_needing_followup = db.query(Lead).filter(
        Lead.status.in_([LeadStatus.CONTACTED.value, LeadStatus.NURTURING.value]),
        Lead.last_contact <= three_days_ago
    ).all()
    
    for lead in leads_needing_followup:
        # Schedule follow-up task
        schedule_followup(lead.id, 1, db)  # Assuming user_id 1 is the default user
        
        # For leads with high scores, also send an automated email
        if lead.score >= 75:
            send_automated_email(lead.id, "high_priority_followup")
    
    logger.info(f"Processed follow-ups for {len(leads_needing_followup)} leads")

# ------------------ Scheduled Tasks ------------------

@app.on_event("startup")
async def startup_event():
    # Start background task scheduler
    asyncio.create_task(background_task_scheduler())

async def background_task_scheduler():
    """
    Schedule and run periodic background tasks.
    """
    while True:
        # Create a new database session for the background tasks
        db = SessionLocal()
        
        try:
            # Run lead scoring update (every 6 hours)
            await update_all_lead_scores(db)
            
            # Send task reminders (daily)
            await send_task_reminders(db)
            
            # Process lead follow-ups (daily)
            await process_lead_followups(db)
            
        except Exception as e:
            logger.error(f"Error in background tasks: {str(e)}")
        
        finally:
            db.close()
        
        # Wait for 6 hours before running again
        await asyncio.sleep(6 * 60 * 60)

# ------------------ Main Entry Point ------------------

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "AI Lead Management System API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)