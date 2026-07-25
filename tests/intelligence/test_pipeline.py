"""Test IntelligencePipeline.""" 

from unittest.mock import Mock
from labi.intelligence.pipeline import IntelligencePipeline, IntelligenceResult
from labi.decision.types import ExecutionAction

def test_pipeline_orchestration():
    mock_classifier = Mock()
    mock_classifier.classify.return_value = Mock(required_role="answer")
    mock_router = Mock()
    mock_router.decide.return_value = Mock(action="reuse")
    mock_engine = Mock()
    mock_engine.decide.return_value = Mock(action=ExecutionAction.EXECUTE_MEMORY)
    pipeline = IntelligencePipeline(mock_classifier, mock_router, mock_engine)
    result = pipeline.process("test goal")
    assert isinstance(result, IntelligenceResult)
    assert result.goal == "test goal"
    mock_classifier.classify.assert_called_once()
