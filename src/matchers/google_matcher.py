import numpy as np
import asyncio
import openai
from src.config import CONCURRENCY
from loguru import logger

async def activity_confidence_check(batch_df):
    """
    Compute dynamic, description-aware Google activity filtering.
    Returns a list of booleans indicating which rows are 'active enough' to keep.
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
    def google_activity_score(row, weight=1.0):
        reviews = row.get("Reviews count", 0) or 0
        rating = row.get("Reviews rating", 0) or 0
        photos = row.get("Photo count", 0) or 0

        review_score = np.log1p(reviews) / np.log1p(1000)
        photo_score = np.log1p(photos) / np.log1p(200)
        rating_score = (rating - 2.5) / 2.5 if rating else 0

        total = 0.5 * review_score + 0.3 * photo_score + 0.2 * rating_score
        return total / weight

    # Run LLM weighting phase
    descriptions = batch_df.get("Description 1", [""] * len(batch_df)).tolist()
    weights = [1.0] * len(batch_df)  # Default neutral weights

    # Fetch descriptions
    descriptions = batch_df.get("Description 1", [""] * len(batch_df)).tolist()

    # Run LLM calls
    llm_weights = await llm_expected_visibility_weight(descriptions)

    # Update only where valid weights are returned
    for i, w in enumerate(llm_weights):
        if isinstance(w, (int, float)) and 0.1 <= w <= 5.0:
            weights[i] = w

    # Compute normalized activity scores
    scores = []
    j = 0
    for row in batch_df.iterrows():
        score = google_activity_score(row, weight=weights[j])
        j+=1
        scores.append(score)

    # Adaptive threshold
    scores = np.array(scores)
    threshold = max(0.2, np.mean(scores) - 0.3 * np.std(scores))

    # Return boolean flags
    keep_flags = (scores >= threshold).tolist()

    logger.debug(f"📊 Activity filter — dynamic threshold={threshold:.2f}, kept {sum(keep_flags)}/{len(keep_flags)}")
    return keep_flags
