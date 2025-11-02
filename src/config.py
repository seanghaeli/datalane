# src/config.py
from dotenv import load_dotenv
import os

load_dotenv()

# API Keys
ZYTE_API_KEY = os.getenv("ZYTE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Runtime parameters
BATCH_SIZE = 15
FUZZY_THRESHOLD = 50
CONCURRENCY = 100
LOG_LEVEL = "DEBUG"

# URLs
GOV_URL = "https://rceapi.estado.pr.gov/api/corporation/search"
ZYTE_URL = "https://api.zyte.com/v1/extract"

# File names
INPUT_CSV = "businesses.csv"
OUTPUT_CSV = "businesses_to_keep.csv"