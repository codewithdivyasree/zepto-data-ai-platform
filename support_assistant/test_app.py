import os

os.environ["MOCK_LLM"] = "1"

from main import classify_intent, direct_answer


def test_policy_routing():
    assert classify_intent({"query": "What is the delivery fee?"})["intent"] == "policy_question"


def test_general_routing_and_schema_fields():
    state = direct_answer({"query": "Who invented Python?"})
    assert state["sources"] == []
    assert state["confidence"] == 1.0

