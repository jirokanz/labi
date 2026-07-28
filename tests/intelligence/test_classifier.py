from labi.intelligence.classifier import TaskClassifier


def _classify(goal):
    return TaskClassifier().classify(goal)


def test_requires_web_false_by_default_for_coding_tasks():
    profile = _classify("write a Python function to reverse a linked list")
    assert profile.requires_web is False
    assert profile.web_confidence == 0.0


def test_requires_web_true_and_high_confidence_for_freshness_keywords():
    profile = _classify("what's the latest version of PostgreSQL?")
    assert profile.requires_web is True
    assert profile.web_confidence == 0.9


def test_requires_web_true_and_lower_confidence_for_role_holder_pattern():
    profile = _classify("who is the CEO of OpenAI?")
    assert profile.requires_web is True
    assert profile.web_confidence == 0.75


def test_requires_web_false_for_role_pattern_with_historical_marker():
    profile = _classify("who was the first president of the United States?")
    assert profile.requires_web is False
    assert profile.web_confidence == 0.0


def test_requires_web_matches_current_or_new_qualifier_before_role():
    profile = _classify("who is the new governor of California?")
    assert profile.requires_web is True


def test_requires_web_does_not_fire_on_role_word_without_who_is_pattern():
    # Mentioning a role title isn't itself a signal -- only the
    # "who is the <role>" question pattern is.
    profile = _classify("write a biography of a fictional king")
    assert profile.requires_web is False


def test_requires_web_false_for_plural_role_without_current_who_pattern():
    profile = _classify("explain how a monarchy works")
    assert profile.requires_web is False
