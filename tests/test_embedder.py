"""Tests for rag.embedder — text chunking and embedding (OpenAI mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag.embedder import Embedder, chunk_text


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        chunks = chunk_text("hello world", size=100, overlap=10)
        assert chunks == ["hello world"]

    def test_overlap(self) -> None:
        text = "A" * 100
        chunks = chunk_text(text, size=40, overlap=10)
        # Each chunk starts 30 chars after the previous
        assert len(chunks) == 4  # 0-40, 30-70, 60-100, 90-100
        # Overlapping region should be identical
        assert chunks[0][-10:] == chunks[1][:10]

    def test_empty_text(self) -> None:
        assert chunk_text("") == []

    def test_exact_size(self) -> None:
        text = "A" * 50
        chunks = chunk_text(text, size=50, overlap=10)
        # First chunk covers full text; overlap causes a second small chunk
        assert chunks[0] == text
        assert len(chunks) >= 1


class TestEmbedder:
    @patch("rag.embedder.OpenAI")
    def test_embed_calls_api(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Simulate API response
        mock_item_1 = MagicMock()
        mock_item_1.embedding = [0.1] * 10
        mock_item_2 = MagicMock()
        mock_item_2.embedding = [0.2] * 10

        mock_response = MagicMock()
        mock_response.data = [mock_item_1, mock_item_2]
        mock_response.usage.prompt_tokens = 50
        mock_client.embeddings.create.return_value = mock_response

        embedder = Embedder()
        result = embedder.embed(["text one", "text two"])

        assert len(result) == 2
        assert result[0] == [0.1] * 10
        mock_client.embeddings.create.assert_called_once()

    @patch("rag.embedder.OpenAI")
    def test_embed_empty_list(self, mock_openai_cls: MagicMock) -> None:
        embedder = Embedder()
        assert embedder.embed([]) == []
