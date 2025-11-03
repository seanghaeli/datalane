import numpy as np
from src.models import BusinessRecord
from src.clients import OpenAIClient
from loguru import logger

async def _get_expected_visibility_weight(description: str) -> float:
    """
    Get expected visibility weight for a business description using LLM.
    
    Args:
        description: Business description string.
        
    Returns:
        float: Expected visibility weight (0.3 to 2.5).
    """
    system_prompt = (
        "You are helping normalize online activity metrics across business types. "
        "Given a short business description, output a single float weight representing "
        "how much online visibility (Google reviews/photos) you would *expect* the business to have. "
        "Restaurants, bars, hotels ≈ 1.8; retail ≈ 1.2; clinics or salons ≈ 1.0; "
        "mechanics or professional offices ≈ 0.6; industrial or B2B ≈ 0.4. "
        "Only respond with the number, nothing else."
    )
    
    # Cache here - Cache expected visibility weights by description string
    # Same business description should produce same expected visibility weight (low temperature)
    try:
        openai_client = OpenAIClient()
        r = await openai_client.chat_completions_create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Description: {description or 'Unknown'}\nExpected weight:"},
            ],
            temperature=0.2,
            max_tokens=10,
        )
        text = r.choices[0].message.content.strip()
        val = float(text)
        return max(0.3, min(val, 2.5))
    except Exception:
        return 1.0

def _parse_photos_count(value) -> int:
    """Parse photos count from various formats."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        # Treat NaN as 0
        if np.isnan(value):
            return 0
        return int(value) if float(value).is_integer() else int(round(float(value)))
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if s.endswith("+"):
            s = s[:-1]
        return int(s) if s.isdigit() else 0
    return 0

def _google_activity_score(record: BusinessRecord, weight: float = 1.0) -> float:
    """Compute normalized Google activity score for a single record."""
    reviews = record.reviews_count or 0
    rating = record.reviews_rating or 0
    photos = _parse_photos_count(record.photos_count or 0)
    
    # Normalize features into [0,1] using conservative caps
    norm_reviews = min(max(reviews, 0), 300) / 300.0
    norm_photos = min(max(photos, 0), 100) / 100.0
    norm_rating = 0.0
    if rating:
        norm_rating = max(0.0, min((rating - 3.0) / 2.0, 1.0))

    # Weighted blend
    activity = 0.6 * norm_reviews + 0.25 * norm_photos + 0.15 * norm_rating

    # Adjust by expected visibility weight and clamp
    adjusted = activity / max(weight, 0.1)
    return float(max(0.0, min(adjusted, 1.0)))

async def activity_confidence_check(record: BusinessRecord) -> int:
    """
    Compute dynamic, description-aware Google activity filtering for a single record.
    
    Args:
        record (BusinessRecord): Business record with Google activity data.
    
    Returns:
        int: -1 indicates a pronounced lack of activity relative to expected category importance; otherwise 0.
    """
    # Build description string
    desc = (record.description_1 or "").strip()
    mt = (record.main_type or "").strip()
    combo = f"{mt} {desc}".strip() if (mt or desc) else "Unknown"
    
    # Get expected visibility weight
    weight = await _get_expected_visibility_weight(combo)
    if not (0.1 <= weight <= 5.0):
        weight = 1.0
    
    # Compute score
    score = _google_activity_score(record, weight=weight)
    
    # Threshold: <= 0.2 => -1 (pronounced lack of activity), else 0
    LOWER = 0.2
    return -1 if score <= LOWER else 0
