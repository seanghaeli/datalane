import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from src.models import BusinessRecord, CandidateRecord, MatchingResult
from src.matchers.matching_orchestrator import matching_orchestrator

@pytest.mark.asyncio
async def test_matching_orchestrator_simpler_mock_pattern():
    """
    Simple mocking pattern using patch decorators.
    This tests running the matching orchestrator with a single record and a single candidate.
    """
    test_business = BusinessRecord(
        name="Simple Test Business",
        street_1="789 Test Rd",
        description_1="Test business",
        main_type="Test",
        reviews_count=10,
        reviews_rating=3.5,
        photos_count="5",
    )
    
    test_candidates = [
        CandidateRecord(name="Simple Test Business", address="789 Test Rd"),
    ]
    
    # Create async mock functions
    mock_llm_response = MagicMock()
    mock_llm_response.choices = [MagicMock()]
    mock_llm_response.choices[0].message.content = "YES"
    
    mock_weight_response = MagicMock()
    mock_weight_response.choices = [MagicMock()]
    mock_weight_response.choices[0].message.content = "1.0"
    
    # Patch at the point of use (in the modules that import the clients)
    with patch("src.matchers.llm_matcher.OpenAIClient") as mock_openai_llm, \
         patch("src.matchers.google_matcher.OpenAIClient") as mock_openai_google:
        
        # Set up the mock instances
        mock_openai_llm_instance = MagicMock()
        mock_openai_llm_instance.chat_completions_create = AsyncMock(
            return_value=mock_llm_response
        )
        mock_openai_llm.return_value = mock_openai_llm_instance
        
        mock_openai_google_instance = MagicMock()
        mock_openai_google_instance.chat_completions_create = AsyncMock(
            return_value=mock_weight_response
        )
        mock_openai_google.return_value = mock_openai_google_instance
        
        # Reset singleton state
        from src.clients import openai_client as oai_client_module
        oai_client_module.OpenAIClient._instance = None
        oai_client_module.OpenAIClient._initialized = False
        
        # Run the test with single record
        result = await matching_orchestrator(test_business, test_candidates)
        
        # Verify results
        assert isinstance(result, MatchingResult)
        assert result.name == "Simple Test Business"
        assert result.results_llm is True
        
        # Verify mocks were called
        assert mock_openai_llm_instance.chat_completions_create.called

