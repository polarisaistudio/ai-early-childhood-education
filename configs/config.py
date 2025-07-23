"""
Configuration settings for AI Early Childhood Education System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

# Child Development Constants
AGE_RANGES = {
    "toddler": (12, 36),           # 1-3 years in months
    "preschool": (36, 60),         # 3-5 years
    "early_elementary": (60, 96)   # 5-8 years
}

ATTENTION_SPANS = {
    "toddler": (2, 5),             # 2-5 minutes
    "preschool": (10, 15),         # 10-15 minutes  
    "early_elementary": (15, 25)   # 15-25 minutes
}

LEARNING_STYLES = ["visual", "auditory", "kinesthetic", "mixed"]

DEVELOPMENTAL_PACES = ["slow", "typical", "fast"]

INTERESTS = [
    "animals", "vehicles", "music", "stories", "building", 
    "art", "nature", "sports", "cooking", "dancing",
    "puzzles", "science", "numbers", "letters"
]

# Content Generation Settings
DEFAULT_MODEL = "gpt-3.5-turbo"
CONTENT_TEMPERATURE = 0.7
MAX_TOKENS = 1000

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
