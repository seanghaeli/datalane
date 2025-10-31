import numpy as np
import asyncio
import openai
from src.config import CONCURRENCY
from loguru import logger

async def activity_confidence_check(batch_df):
    """
    Compute dynamic, description-aware Google activity filtering.
    Returns a list of ints where -1 indicates a pronounced lack of activity
    relative to expected category importance; otherwise 0.
    """

    # Helper 1: LLM weighting
    async def llm_expected_visibility_weight(descriptions: list[str]) -> list[float]:
        system_prompt = (
            "You are helping normalize online activity metrics across business types. "
            "Given a short business description, output a single float weight representing "
            "how much online visibility (Google reviews/photos) you would *expect* the business to have. "
            "Restaurants, bars, hotels ≈ 1.8; retail ≈ 1.2; clinics or salons ≈ 1.0; "
            "mechanics or professional offices ≈ 0.6; industrial or B2B ≈ 0.4. "
            "Only respond with the number, nothing else."
        )

        async def one(desc):
            try:
                r = await openai.ChatCompletion.acreate(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Description: {desc or 'Unknown'}\nExpected weight:"},
                    ],
                    temperature=0.2,
                    max_tokens=10,
                )
                text = r["choices"][0]["message"]["content"].strip()
                val = float(text)
                return max(0.3, min(val, 2.5))
            except Exception:
                return 1.0

        tasks = [one(d) for d in descriptions]
        return await asyncio.gather(*tasks)

    # Helper 2: score computation
    def _parse_photos_count(value) -> int:
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

    def google_activity_score(row, weight=1.0):
        reviews = row.get("Reviews count", 0) or 0
        rating = row.get("Reviews rating", 0) or 0
        photos = _parse_photos_count(row.get("Photos count", 0))
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

    # Run LLM weighting phase
    descriptions = batch_df.get("Description 1", [""] * len(batch_df)).tolist()
    main_types = batch_df.get("Main type", [""] * len(batch_df)).tolist()
    weights = [1.0] * len(batch_df)  # Default neutral weights

    # Build simple "category + description" strings; either part may be empty
    llm_inputs = []
    for desc, mt in zip(descriptions, main_types):
        if isinstance(desc, (float, np.floating)) and np.isnan(desc):
            desc = ""
        if isinstance(mt, (float, np.floating)) and np.isnan(mt):
            mt = ""
        desc = (desc or "").strip()
        mt = (mt or "").strip()
        combo = f"{mt} {desc}".strip() if (mt or desc) else "Unknown"
        llm_inputs.append(combo)

    # Run LLM calls
    llm_weights = await llm_expected_visibility_weight(llm_inputs)

    # Update only where valid weights are returned
    for i, w in enumerate(llm_weights):
        if isinstance(w, (int, float)) and 0.1 <= w <= 5.0:
            weights[i] = w

    # Compute normalized activity scores
    scores = []
    for i, (_, row) in enumerate(batch_df.iterrows()):
        score = google_activity_score(row, weight=weights[i])
        scores.append(score)

    # Threshold: <= 0.2 => -1 (pronounced lack of activity), else 0
    LOWER = 0.2
    scores = np.array(scores)
    out = []
    for s in scores.tolist():
        if s <= LOWER:
            out.append(-1)
        else:
            out.append(0)
    logger.debug(
        f"📊 Activity filter — LOWER={LOWER:.2f}, 0={out.count(0)}, -1={out.count(-1)}"
    )
    return out
