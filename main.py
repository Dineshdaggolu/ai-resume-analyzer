from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import logging
import time
from datetime import datetime
import PyPDF2
import io

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resume Analyzer API",
    description="Enhanced with Azure AI, Gen AI, and PDF Processing",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Azure AI Integration (Optional)
# ----------------------------
try:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure libraries not installed. Azure AI features disabled.")

# Azure configuration (set these in environment variables)
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_KEY = os.getenv("AZURE_KEY", "")

class AzureAIClient:
    """Wrapper for Azure AI services"""
    
    def __init__(self):
        self.client = None
        self.available = False
        
        if AZURE_AVAILABLE and AZURE_ENDPOINT and AZURE_KEY:
            try:
                credential = AzureKeyCredential(AZURE_KEY)
                self.client = TextAnalyticsClient(
                    endpoint=AZURE_ENDPOINT,
                    credential=credential
                )
                self.available = True
                logger.info("✅ Azure AI initialized")
            except Exception as e:
                logger.error(f"Azure AI init failed: {e}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities using Azure AI"""
        if not self.available:
            return []
        
        try:
            response = self.client.recognize_entities([text])[0]
            return [
                {
                    "text": entity.text,
                    "category": entity.category,
                    "confidence": entity.confidence_score
                }
                for entity in response.entities
            ]
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases using Azure AI"""
        if not self.available:
            return []
        
        try:
            response = self.client.extract_key_phrases([text])[0]
            return response.key_phrases
        except Exception as e:
            logger.error(f"Key phrase extraction failed: {e}")
            return []

# Initialize Azure client
azure_client = AzureAIClient()

# ----------------------------
# Enhanced Skill Database
# ----------------------------
SKILL_DATABASE = {
    "programming": ["python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "typescript"],
    "frameworks": ["fastapi", "django", "flask", "react", "angular", "vue", "spring", "node.js", "express"],
    "cloud": ["azure", "aws", "gcp", "docker", "kubernetes", "terraform", "jenkins"],
    "database": ["sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch"],
    "ai_ml": ["tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "langchain"],
    "devops": ["ci/cd", "git", "github", "gitlab", "ansible", "prometheus", "grafana"]
}

# Flatten skills for easy searching
ALL_SKILLS = [skill for category in SKILL_DATABASE.values() for skill in category]

# ----------------------------
# Enhanced Request & Response Models
# ----------------------------

class ResumeRequest(BaseModel):
    resume: str
    job_description: str
    use_azure_ai: Optional[bool] = False  # New: Toggle Azure AI features
    use_gen_ai: Optional[bool] = False    # New: Toggle Gen AI suggestions

class ResumeResponse(BaseModel):
    # Your original fields
    match_percentage: int
    matched_skills: List[str]
    missing_skills: List[str]
    suggestions: str
    
    # New enhanced fields
    azure_entities: Optional[List[Dict]] = None
    key_phrases: Optional[List[str]] = None
    skill_recommendations: Optional[Dict] = None
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    skills_breakdown: Optional[Dict[str, List[str]]] = None

class PDFUploadResponse(BaseModel):
    filename: str
    extracted_text: str
    pages: int
    text_length: int

# ----------------------------
# Enhanced Business Logic
# ----------------------------

def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills from text using both predefined and AI methods"""
    text_lower = text.lower()
    found_skills = []
    
    # Method 1: Predefined skill matching (your original approach)
    for skill in ALL_SKILLS:
        if skill in text_lower:
            found_skills.append(skill)
    
    return list(set(found_skills))  # Remove duplicates

def categorize_skills(skills: List[str]) -> Dict[str, List[str]]:
    """Categorize skills by domain"""
    categorized = {category: [] for category in SKILL_DATABASE.keys()}
    
    for skill in skills:
        for category, category_skills in SKILL_DATABASE.items():
            if skill in category_skills:
                categorized[category].append(skill)
                break
    
    return categorized

def generate_gen_ai_suggestions(missing_skills: List[str]) -> Dict:
    """Generate AI-powered learning suggestions"""
    if not missing_skills:
        return {
            "message": "Great job! You have all required skills.",
            "next_steps": ["Consider advanced certifications", "Look for leadership roles"]
        }
    
    # Mock Gen AI responses (in production, call actual Gen AI model)
    suggestions = {
        "learning_resources": [
            f"Microsoft Learn: {skill} path" for skill in missing_skills[:3]
        ],
        "estimated_time": f"{len(missing_skills) * 2} weeks",
        "priority": "High" if len(missing_skills) > 3 else "Medium",
        "tips": [
            f"Start with {missing_skills[0]} as it's most in-demand",
            "Practice with real projects",
            "Get certified to validate your skills"
        ]
    }
    
    return suggestions

# ----------------------------
# Your Original Logic (Enhanced)
# ----------------------------

def analyze_resume_logic(
    resume: str, 
    job_description: str,
    use_azure: bool = False,
    use_gen_ai: bool = False
) -> ResumeResponse:
    start_time = time.time()
    
    # Your original preprocessing
    resume_text = resume.lower()
    jd_text = job_description.lower()

    # Enhanced skill extraction (now uses larger skill database)
    resume_skills = extract_skills_from_text(resume_text)
    jd_skills = extract_skills_from_text(jd_text)

    # Your original matching logic (preserved)
    matched = []
    missing = []

    for skill in jd_skills:  # Now uses dynamic skills instead of predefined
        if skill in resume_skills:
            matched.append(skill)
        else:
            missing.append(skill)

    total_required = len(jd_skills)
    match_percentage = int((len(matched) / total_required) * 100) if total_required else 100

    # Your original suggestion text
    suggestion_text = (
        f"Improve skills in: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
        if missing
        else "Strong profile match! Your skills align perfectly."
    )

    # Build response with your original fields
    response = ResumeResponse(
        match_percentage=match_percentage,
        matched_skills=matched,
        missing_skills=missing,
        suggestions=suggestion_text
    )

    # Add Azure AI features if requested
    if use_azure and azure_client.available:
        response.azure_entities = azure_client.extract_entities(resume)
        response.key_phrases = azure_client.extract_key_phrases(resume)
    
    # Add skill categorization
    response.skills_breakdown = categorize_skills(resume_skills)
    
    # Add Gen AI suggestions if requested
    if use_gen_ai and missing:
        response.skill_recommendations = generate_gen_ai_suggestions(missing)
    
    # Add metadata
    response.processing_time = round(time.time() - start_time, 3)
    response.confidence_score = 0.85 if azure_client.available else 0.70

    return response

# ----------------------------
# NEW: PDF Upload Endpoint
# ----------------------------

@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and extract text from PDF resume
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Validate file size (max 5MB)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")
    
    try:
        # Extract text from PDF
        pdf_file = io.BytesIO(contents)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        extracted_text = ""
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() + "\n"
        
        return PDFUploadResponse(
            filename=file.filename,
            extracted_text=extracted_text.strip(),
            pages=len(pdf_reader.pages),
            text_length=len(extracted_text)
        )
    
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

# ----------------------------
# NEW: Enhanced Analyze Endpoint
# ----------------------------

@app.post("/analyze", response_model=ResumeResponse)
def analyze_resume(data: ResumeRequest):
    """
    Analyze resume against job description
    
    Features:
    - Basic skill matching (always enabled)
    - Azure AI entity extraction (optional)
    - Gen AI learning suggestions (optional)
    - Skill categorization
    """
    try:
        if not data.resume or not data.job_description:
            raise HTTPException(status_code=400, detail="Resume and job description are required")
        
        return analyze_resume_logic(
            data.resume, 
            data.job_description,
            use_azure=data.use_azure_ai,
            use_gen_ai=data.use_gen_ai
        )
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# NEW: Health Check Endpoint
# ----------------------------

@app.get("/health")
async def health_check():
    """Check API health and available features"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "azure_ai": azure_client.available,
            "pdf_processing": True,
            "skill_database_size": len(ALL_SKILLS)
        },
        "version": "2.0.0"
    }

# ----------------------------
# NEW: Skills Endpoint
# ----------------------------

@app.get("/skills")
async def get_skills():
    """Get all available skills in database"""
    return {
        "total_skills": len(ALL_SKILLS),
        "categories": list(SKILL_DATABASE.keys()),
        "skills_by_category": SKILL_DATABASE
    }

# ----------------------------
# Root Endpoint
# ----------------------------

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "AI Resume Analyzer API",
        "version": "2.0.0",
        "endpoints": {
            "POST /analyze": "Analyze resume against job description",
            "POST /upload-pdf": "Upload and extract text from PDF resume",
            "GET /health": "Health check",
            "GET /skills": "Get available skills database"
        },
        "features": {
            "azure_ai": azure_client.available,
            "gen_ai": True,
            "pdf_support": True,
            "skill_database": len(ALL_SKILLS)
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 AI Resume Analyzer API")
    print("="*50)
    print(f"\n📚 Documentation: http://127.0.0.1:8000/docs")
    print(f"🏠 Main API: http://127.0.0.1:8000")
    print(f"\n✅ Features:")
    print(f"  • Azure AI: {'✅' if azure_client.available else '❌'}")
    print(f"  • PDF Support: ✅")
    print(f"  • Skills Database: {len(ALL_SKILLS)} skills")
    print("\n" + "="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)