"""
Typed data models for business matching pipeline.
All data structures used throughout the codebase should be defined here.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class BusinessRecord:
    """Input business record loaded from CSV."""
    name: str
    street_1: Optional[str] = None
    description_1: Optional[str] = None
    main_type: Optional[str] = None
    reviews_count: Optional[int] = None
    reviews_rating: Optional[float] = None
    photos_count: Optional[str] = None  # Can be string like "708+" or int


@dataclass
class CandidateRecord:
    """Registry candidate record from government database."""
    name: str
    address: Optional[str] = None


@dataclass
class MatchingResult:
    """Final matching result for a business."""
    name: str
    results: bool  # Classical fuzzy match result
    results_llm: bool  # LLM match result
    results_google_check: int  # Google activity check: -1 (low activity) or 0 (otherwise)
    overall_results: bool  # Final combined decision
