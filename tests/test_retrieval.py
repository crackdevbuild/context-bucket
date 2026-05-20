"""Unit tests for context_bucket/retrieval.py"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from context_bucket.retrieval import (
    lexical_tokens,
    semantic_terms,
    stem_token,
    char_ngrams,
    cosine_similarity,
    lexical_overlap_score,
    set_overlap_score,
    scope_priority_bonus,
    selection_reason,
    dedupe_context_items,
    record_rank_bonus,
    token_set_similarity,
    record_visible,
    record_summary_stale_for_context,
    policy_exclusion_reason,
)
from context_bucket.models import (
    ContextBucketContextItem,
    ContextBucketRetrieveRequest,
    ContextBucketRecord,
)


class TestLexicalTokens:
    """Tests for lexical_tokens function."""

    def test_simple_text(self) -> None:
        """Test tokenizing simple text."""
        tokens = lexical_tokens("hello world")
        assert tokens == ["hello", "world"]

    def test_case_normalization(self) -> None:
        """Test that tokens are lowercased."""
        tokens = lexical_tokens("Hello WORLD")
        assert tokens == ["hello", "world"]

    def test_short_tokens_included(self) -> None:
        """Test that short tokens are included when matched by the tokenizer."""
        tokens = lexical_tokens("a an the hello")
        assert tokens == ["an", "hello", "the"]

    def test_alphanumeric(self) -> None:
        """Test that alphanumeric tokens are included."""
        tokens = lexical_tokens("test123 abc456")
        assert "test123" in tokens
        assert "abc456" in tokens

    def test_punctuation_stripped(self) -> None:
        """Test that punctuation is stripped."""
        tokens = lexical_tokens("hello, world! test.")
        assert tokens == ["hello", "test", "world"]

    def test_empty_string(self) -> None:
        """Test empty string returns empty list."""
        tokens = lexical_tokens("")
        assert tokens == []

    def test_duplicates_removed(self) -> None:
        """Test that duplicates are removed and sorted."""
        tokens = lexical_tokens("hello hello world world")
        assert tokens == ["hello", "world"]


class TestSemanticTerms:
    """Tests for semantic_terms function."""

    def test_basic_terms(self) -> None:
        """Test basic semantic term extraction."""
        terms = semantic_terms("brief summary")
        assert "brief" in terms
        assert "summary" in terms

    def test_semantic_expansions(self) -> None:
        """Test that semantic expansions are included."""
        terms = semantic_terms("brief")
        # "brief" should expand to include "concise", "short"
        assert "brief" in terms
        assert "concise" in terms
        assert "short" in terms

    def test_char_ngrams_included(self) -> None:
        """Test that character ngrams are included."""
        terms = semantic_terms("hello")
        # Should include char ngrams
        assert any(term.startswith("^") for term in terms)


class TestStemToken:
    """Tests for stem_token function."""

    def test_no_stem_needed(self) -> None:
        """Test short words that don't need stemming."""
        assert stem_token("cat") == "cat"

    def test_ing_suffix(self) -> None:
        """Test -ing suffix removal."""
        assert stem_token("running") == "runn"

    def test_ed_suffix(self) -> None:
        """Test -ed suffix removal."""
        assert stem_token("tested") == "test"

    def test_ies_suffix(self) -> None:
        """Test -ies suffix conversion to y."""
        assert stem_token("parties") == "party"

    def test_ied_suffix(self) -> None:
        """Test -ied suffix conversion to y."""
        assert stem_token("carried") == "carry"

    def test_s_suffix(self) -> None:
        """Test -s suffix removal."""
        assert stem_token("cats") == "cat"

    def test_es_suffix(self) -> None:
        """Test -es suffix removal."""
        assert stem_token("boxes") == "box"

    def test_ment_suffix(self) -> None:
        """Test -ment suffix removal."""
        assert stem_token("development") == "develop"

    def test_tion_suffix(self) -> None:
        """Test -tion suffix removal."""
        assert stem_token("creation") == "crea"


class TestCharNgrams:
    """Tests for char_ngrams function."""

    def test_short_token(self) -> None:
        """Test ngrams for short tokens."""
        ngrams = char_ngrams("ab", 4)
        assert ngrams == ["^ab$"]

    def test_normal_token(self) -> None:
        """Test ngrams for normal length tokens."""
        ngrams = char_ngrams("hello", 4)
        assert "^hel" in ngrams
        assert "hell" in ngrams
        assert "ello" in ngrams
        assert "llo$" in ngrams

    def test_single_char(self) -> None:
        """Test ngrams for single char."""
        ngrams = char_ngrams("a", 4)
        assert ngrams == ["^a$"]


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        """Test similarity of identical vectors."""
        sim = cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert sim == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Test similarity of orthogonal vectors."""
        sim = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert sim == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Test similarity of opposite vectors (clamped to 0)."""
        sim = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert sim == pytest.approx(0.0)

    def test_empty_vectors(self) -> None:
        """Test with empty vectors."""
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], []) == 0.0

    def test_different_lengths(self) -> None:
        """Test with different length vectors."""
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0


class TestLexicalOverlapScore:
    """Tests for lexical_overlap_score function."""

    def test_identical_tokens(self) -> None:
        """Test score for identical tokens."""
        score = lexical_overlap_score(["a", "b"], ["a", "b"])
        assert score == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Test score for no overlap."""
        score = lexical_overlap_score(["a", "b"], ["c", "d"])
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Test score for partial overlap."""
        score = lexical_overlap_score(["a", "b"], ["b", "c"])
        # overlap = 1, sqrt(2*2) = 2, so 1/2 = 0.5
        assert score == pytest.approx(0.5)

    def test_empty_tokens(self) -> None:
        """Test score with empty tokens."""
        assert lexical_overlap_score([], ["a"]) == 0.0
        assert lexical_overlap_score(["a"], []) == 0.0


class TestSetOverlapScore:
    """Tests for set_overlap_score function."""

    def test_identical_sets(self) -> None:
        """Test score for identical sets."""
        score = set_overlap_score(["a", "b"], ["a", "b"])
        assert score == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Test score for no overlap."""
        score = set_overlap_score(["a", "b"], ["c", "d"])
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Test score for partial overlap."""
        score = set_overlap_score(["a", "b"], ["b", "c"])
        assert score == pytest.approx(0.5)


class TestScopePriorityBonus:
    """Tests for scope_priority_bonus function."""

    def test_session_scope(self) -> None:
        """Test session scope gets highest bonus."""
        assert scope_priority_bonus("session") == 0.12

    def test_user_scope(self) -> None:
        """Test user scope gets medium bonus."""
        assert scope_priority_bonus("user") == 0.08

    def test_global_scope(self) -> None:
        """Test global scope gets lowest bonus."""
        assert scope_priority_bonus("global") == 0.03


class TestSelectionReason:
    """Tests for selection_reason function."""

    def test_semantic_match(self) -> None:
        """Test semantic match reason."""
        reason = selection_reason("session", 0.8, 0.3, 0.0)
        assert reason == "session_scope_semantic_match"

    def test_lexical_match(self) -> None:
        """Test lexical match reason."""
        reason = selection_reason("user", 0.2, 0.7, 0.0)
        assert reason == "user_scope_lexical_match"

    def test_keyword_match(self) -> None:
        """Test keyword match reason."""
        reason = selection_reason("global", 0.0, 0.0, 0.1)
        assert reason == "global_scope_keyword_match"

    def test_lexical_keyword_match(self) -> None:
        """Test lexical with keyword match reason."""
        reason = selection_reason("session", 0.0, 0.5, 0.1)
        assert reason == "session_scope_lexical_keyword_match"


class TestDedupeContextItems:
    """Tests for dedupe_context_items function."""

    def test_no_duplicates(self) -> None:
        """Test deduping with no duplicates."""
        now = datetime.now(timezone.utc)
        items = [
            ContextBucketContextItem(
                record_id="r1",
                chunk_id="c1",
                kind="research_finding",
                scope="session",
                text="text1",
                created_at=now,
            ),
            ContextBucketContextItem(
                record_id="r2",
                chunk_id="c2",
                kind="research_finding",
                scope="session",
                text="text2",
                created_at=now,
            ),
        ]
        deduped = dedupe_context_items(items)
        assert len(deduped) == 2

    def test_with_duplicates(self) -> None:
        """Test deduping removes duplicates."""
        now = datetime.now(timezone.utc)
        items = [
            ContextBucketContextItem(
                record_id="r1",
                chunk_id="c1",
                kind="research_finding",
                scope="session",
                text="text1",
                created_at=now,
            ),
            ContextBucketContextItem(
                record_id="r1",
                chunk_id="c1",
                kind="research_finding",
                scope="session",
                text="text1",
                created_at=now,
            ),
        ]
        deduped = dedupe_context_items(items)
        assert len(deduped) == 1

    def test_preserves_order(self) -> None:
        """Test that order is preserved."""
        now = datetime.now(timezone.utc)
        items = [
            ContextBucketContextItem(
                record_id="r1",
                chunk_id="c1",
                kind="research_finding",
                scope="session",
                text="text1",
                created_at=now,
            ),
            ContextBucketContextItem(
                record_id="r2",
                chunk_id="c2",
                kind="research_finding",
                scope="session",
                text="text2",
                created_at=now,
            ),
            ContextBucketContextItem(
                record_id="r1",
                chunk_id="c1",
                kind="research_finding",
                scope="session",
                text="text1",
                created_at=now,
            ),
            ContextBucketContextItem(
                record_id="r3",
                chunk_id="c3",
                kind="research_finding",
                scope="session",
                text="text3",
                created_at=now,
            ),
        ]
        deduped = dedupe_context_items(items)
        assert len(deduped) == 3
        assert deduped[0].record_id == "r1"
        assert deduped[1].record_id == "r2"
        assert deduped[2].record_id == "r3"


class TestRecordRankBonus:
    """Tests for record_rank_bonus function."""

    def test_research_report(self) -> None:
        """Test bonus for research_report kind."""
        record = {"kind": "research_report"}
        assert record_rank_bonus(record) == 0.15

    def test_research_finding(self) -> None:
        """Test bonus for research_finding kind."""
        record = {"kind": "research_finding"}
        assert record_rank_bonus(record) == 0.15

    def test_decision_outcome(self) -> None:
        """Test bonus for decision_outcome kind."""
        record = {"kind": "decision_outcome"}
        assert record_rank_bonus(record) == 0.08

    def test_user_profile_note(self) -> None:
        """Test bonus for user_profile_note kind."""
        record = {"kind": "user_profile_note"}
        assert record_rank_bonus(record) == 0.08

    def test_other_kind(self) -> None:
        """Test no bonus for other kinds."""
        record = {"kind": "search_query"}
        assert record_rank_bonus(record) == 0.0

    def test_record_object(self) -> None:
        """Test with ContextBucketRecord object."""
        record = ContextBucketRecord(
            id="r1",
            kind="research_report",
            text="test",
            created_at=datetime.now(timezone.utc),
        )
        assert record_rank_bonus(record) == 0.15


class TestTokenSetSimilarity:
    """Tests for token_set_similarity function."""

    def test_identical_sets(self) -> None:
        """Test similarity of identical sets."""
        sim = token_set_similarity({"a", "b"}, {"a", "b"})
        assert sim == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Test similarity with no overlap."""
        sim = token_set_similarity({"a", "b"}, {"c", "d"})
        assert sim == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Test similarity with partial overlap."""
        sim = token_set_similarity({"a", "b"}, {"b", "c"})
        # intersection = 1, union = 3, so 1/3
        assert sim == pytest.approx(1.0 / 3.0)

    def test_empty_sets(self) -> None:
        """Test similarity with empty sets."""
        assert token_set_similarity(set(), {"a"}) == 0.0
        assert token_set_similarity({"a"}, set()) == 0.0


class TestRecordVisible:
    """Tests for record_visible function."""

    def test_session_record_visible(self) -> None:
        """Test session record visibility."""
        record = {"scope": "session", "session_id": "s1", "user_id": None}
        assert record_visible(record, session_id="s1", user_id="u1", include_user_scope=True, include_global_scope=True)
        assert not record_visible(record, session_id="s2", user_id="u1", include_user_scope=True, include_global_scope=True)

    def test_user_record_visible(self) -> None:
        """Test user record visibility."""
        record = {"scope": "user", "session_id": None, "user_id": "u1"}
        assert record_visible(record, session_id="s1", user_id="u1", include_user_scope=True, include_global_scope=True)
        assert not record_visible(record, session_id="s1", user_id="u2", include_user_scope=True, include_global_scope=True)
        assert not record_visible(record, session_id="s1", user_id="u1", include_user_scope=False, include_global_scope=True)

    def test_global_record_visible(self) -> None:
        """Test global record visibility."""
        record = {"scope": "global", "session_id": None, "user_id": None}
        assert record_visible(record, session_id=None, user_id=None, include_user_scope=False, include_global_scope=True)
        assert not record_visible(record, session_id=None, user_id=None, include_user_scope=False, include_global_scope=False)

    def test_record_object(self) -> None:
        """Test with ContextBucketRecord object."""
        record = ContextBucketRecord(
            id="r1",
            kind="research_finding",
            scope="session",
            session_id="s1",
            text="test",
            created_at=datetime.now(timezone.utc),
        )
        assert record_visible(record, session_id="s1", user_id=None, include_user_scope=False, include_global_scope=False)


class TestRecordSummaryStaleForContext:
    """Tests for record_summary_stale_for_context function."""

    def test_fresh_record(self) -> None:
        """Test that fresh record is not stale."""
        record = {
            "created_at": datetime.now(timezone.utc),
            "policy": {"freshness_days": 30},
        }
        # Mock service
        class MockService:
            def _query_max_age_days(self, r: dict) -> int:
                return 30
        assert not record_summary_stale_for_context(MockService(), record, None)

    def test_stale_record_with_max_age(self) -> None:
        """Test that old record is stale with max_age_days."""
        record = {
            "created_at": datetime.now(timezone.utc) - timedelta(days=10),
            "policy": {"freshness_days": 5},
        }
        class MockService:
            def _query_max_age_days(self, r: dict) -> int:
                return 30
        assert record_summary_stale_for_context(MockService(), record, 5)

    def test_no_max_age_uses_policy(self) -> None:
        """Test that policy freshness is used when no max_age_days."""
        record = {
            "created_at": datetime.now(timezone.utc) - timedelta(days=10),
            "policy": {"freshness_days": 5},
        }
        class MockService:
            def _query_max_age_days(self, r: dict) -> int:
                return 5
        # Should use service's _query_max_age_days
        assert record_summary_stale_for_context(MockService(), record, None)


class TestPolicyExclusionReason:
    """Tests for policy_exclusion_reason function."""

    def test_no_exclusion_shareable(self) -> None:
        """Test that shareable confidentiality has no exclusion."""
        record = {"policy": {"confidentiality": "shareable"}}
        payload = ContextBucketRetrieveRequest(query_text="test")
        assert policy_exclusion_reason(record, payload) is None

    def test_user_not_allowed(self) -> None:
        """Test exclusion when user not in allowed list."""
        record = {"policy": {"allowed_user_ids": ["u1", "u2"]}, "scope": "user"}
        payload = ContextBucketRetrieveRequest(query_text="test", user_id="u3")
        assert policy_exclusion_reason(record, payload) == "policy_user_not_allowed"

    def test_session_not_allowed(self) -> None:
        """Test exclusion when session not in allowed list."""
        record = {"policy": {"allowed_session_ids": ["s1"]}, "scope": "session"}
        payload = ContextBucketRetrieveRequest(query_text="test", session_id="s2")
        assert policy_exclusion_reason(record, payload) == "policy_session_not_allowed"

    def test_remote_egress_denied(self) -> None:
        """Test exclusion for hosted model target."""
        record = {"policy": {"allow_remote_model_egress": False}}
        payload = ContextBucketRetrieveRequest(query_text="test", model_target="hosted")
        assert policy_exclusion_reason(record, payload) == "policy_remote_egress_denied"

    def test_local_egress_denied(self) -> None:
        """Test exclusion when local egress denied."""
        record = {"policy": {"allow_local_model_egress": False}}
        payload = ContextBucketRetrieveRequest(query_text="test", model_target="local")
        assert policy_exclusion_reason(record, payload) == "policy_local_egress_denied"

    def test_restricted_global_no_context(self) -> None:
        """Test restricted global scope without user/session."""
        record = {"scope": "global", "policy": {"confidentiality": "restricted"}}
        payload = ContextBucketRetrieveRequest(query_text="test")
        assert policy_exclusion_reason(record, payload) == "policy_restricted_context_required"

    def test_private_global_denied(self) -> None:
        """Test private global scope denied."""
        record = {"scope": "global", "policy": {"confidentiality": "private"}}
        payload = ContextBucketRetrieveRequest(query_text="test")
        assert policy_exclusion_reason(record, payload) == "policy_private_global_denied"