"""Tests for copex.json_utils."""

import pytest
from copex.json_utils import extract_json_array


class TestExtractJsonArray:
    """Tests for extract_json_array."""

    def test_direct_json(self):
        assert extract_json_array('[{"role": "lead"}]') == [{"role": "lead"}]

    def test_code_fence_json(self):
        text = 'Here:\n```json\n[{"role": "lead"}]\n```\nDone!'
        assert extract_json_array(text) == [{"role": "lead"}]

    def test_code_fence_no_lang(self):
        text = 'Result:\n```\n[{"a": 1}]\n```'
        assert extract_json_array(text) == [{"a": 1}]

    def test_mixed_text(self):
        text = 'The team:\n[{"role": "lead"}, {"role": "dev"}]\nEnd.'
        assert extract_json_array(text) == [{"role": "lead"}, {"role": "dev"}]

    def test_truncated_recovers_complete_objects(self):
        text = '[{"role": "lead", "name": "L"}, {"role": "dev", "name": "incomplete'
        result = extract_json_array(text)
        assert len(result) == 1
        assert result[0]["role"] == "lead"

    def test_truncated_mid_string(self):
        text = '[{"role": "lead"}, {"role": "dev", "prompt": "You are a developer who'
        result = extract_json_array(text)
        assert len(result) == 1
        assert result[0]["role"] == "lead"

    def test_escaped_quotes(self):
        text = '[{"role": "lead", "prompt": "Say \\"hello\\" to team"}, {"role": "trunc'
        result = extract_json_array(text)
        assert len(result) == 1
        assert "hello" in result[0]["prompt"]

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="Could not extract"):
            extract_json_array("no json here at all")

    def test_empty_array(self):
        assert extract_json_array("[]") == []

    def test_whitespace(self):
        assert extract_json_array("  \n [1, 2, 3] \n  ") == [1, 2, 3]

    def test_plain_string_array(self):
        assert extract_json_array('["a", "b"]') == ["a", "b"]
