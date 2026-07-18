"""
tests/test_results_repository.py
---------------------------------
Round-trip tests for the MongoDB persistence layer, using mongomock so no
real Atlas cluster is required. Run with: uv run pytest tests/
"""

import mongomock
import pytest

from core.models import AnalysisJob, FusedWindow, GestureFeatures, JobStatus, TimeWindow
from core.results_repository import ResultsRepository

COLLECTION = "TedX"


def make_window(start=0.0, end=5.0) -> TimeWindow:
    return TimeWindow(start_s=start, end_s=end)


def make_fused(idx: int) -> FusedWindow:
    return FusedWindow(
        window=make_window(idx * 5.0, (idx + 1) * 5.0),
        gesture=GestureFeatures(
            window=make_window(idx * 5.0, (idx + 1) * 5.0),
            mean_wrist_velocity=1.0,
            max_wrist_displacement=2.0,
            pose_present_ratio=0.9,
        ),
    )


@pytest.fixture
def repo(monkeypatch):
    monkeypatch.setattr("core.results_repository.MongoClient", mongomock.MongoClient)
    return ResultsRepository(uri="mongodb://localhost/", db_name="test_multiarth")


class TestResultsRepository:
    def test_save_and_get_job(self, repo):
        job = AnalysisJob(job_id="abc123", video_path="/videos/talk.mp4", status=JobStatus.DONE)
        repo.save_job(COLLECTION, job, drive_url="https://drive.google.com/file/d/xyz/view",
                      label="Talk 1", dedupe_key="dedupe-1", duration_s=42.0)

        restored = repo.get_job(COLLECTION, "abc123")
        assert restored is not None
        assert restored.job_id == "abc123"
        assert restored.video_path == "/videos/talk.mp4"

    def test_find_by_dedupe_key(self, repo):
        job = AnalysisJob(job_id="abc123", video_path="/videos/talk.mp4", status=JobStatus.DONE)
        assert repo.find_by_dedupe_key(COLLECTION, "dedupe-1") is None

        repo.save_job(COLLECTION, job, drive_url=None, label=None, dedupe_key="dedupe-1", duration_s=None)
        assert repo.find_by_dedupe_key(COLLECTION, "dedupe-1") == "abc123"
        assert repo.find_by_dedupe_key(COLLECTION, "dedupe-2") is None

    def test_save_and_get_fused_windows(self, repo):
        windows = [make_fused(0), make_fused(1), make_fused(2)]
        repo.save_fused_windows(COLLECTION, "abc123", windows)

        restored = repo.get_all_fused(COLLECTION, "abc123")
        assert len(restored) == 3
        assert [w.window.start_s for w in restored] == [0.0, 5.0, 10.0]
        assert restored[0].gesture.mean_wrist_velocity == pytest.approx(1.0)

    def test_save_fused_windows_replaces_previous(self, repo):
        repo.save_fused_windows(COLLECTION, "abc123", [make_fused(0), make_fused(1)])
        repo.save_fused_windows(COLLECTION, "abc123", [make_fused(0)])

        restored = repo.get_all_fused(COLLECTION, "abc123")
        assert len(restored) == 1

    def test_save_and_get_artifacts(self, repo):
        repo.save_artifacts(
            COLLECTION, "abc123",
            wordlist={"words": [{"lemma": "hello", "count": 3, "pos": "INTJ"}]},
            ngrams={"bigrams": [], "trigrams": []},
            collocations={"hello": {}},
            spectrogram=None,
            waveform=None,
        )
        artifacts = repo.get_artifacts(COLLECTION, "abc123")
        assert artifacts["wordlist"]["words"][0]["lemma"] == "hello"
        assert artifacts["collocations"] == {"hello": {}}

    def test_list_videos(self, repo):
        job = AnalysisJob(job_id="abc123", video_path="/videos/talk.mp4", status=JobStatus.DONE)
        repo.save_job(COLLECTION, job, drive_url=None, label="Talk 1", dedupe_key="dedupe-1", duration_s=10.0)

        videos = repo.list_videos(COLLECTION)
        assert len(videos) == 1
        assert videos[0]["label"] == "Talk 1"

    def test_collections_are_isolated(self, repo):
        """Two named corpora must not see each other's videos or dedupe keys."""
        tedx_job = AnalysisJob(job_id="tedx1", video_path="/videos/tedx.mp4", status=JobStatus.DONE)
        yixi_job = AnalysisJob(job_id="yixi1", video_path="/videos/yixi.mp4", status=JobStatus.DONE)

        repo.save_job("TedX", tedx_job, drive_url=None, label="TedX Talk", dedupe_key="dupe", duration_s=10.0)
        repo.save_job("Yixi", yixi_job, drive_url=None, label="Yixi Talk", dedupe_key="dupe", duration_s=10.0)

        tedx_videos = repo.list_videos("TedX")
        yixi_videos = repo.list_videos("Yixi")
        assert [v["_id"] for v in tedx_videos] == ["tedx1"]
        assert [v["_id"] for v in yixi_videos] == ["yixi1"]

        # Same dedupe_key in both corpora resolves to each corpus's own video
        assert repo.find_by_dedupe_key("TedX", "dupe") == "tedx1"
        assert repo.find_by_dedupe_key("Yixi", "dupe") == "yixi1"

        # A job_id shipped only to TedX is invisible from Yixi
        assert repo.get_job("Yixi", "tedx1") is None

    def test_list_collections(self, repo):
        assert repo.list_collections() == []

        tedx_job = AnalysisJob(job_id="tedx1", video_path="/videos/tedx.mp4", status=JobStatus.DONE)
        yixi_job = AnalysisJob(job_id="yixi1", video_path="/videos/yixi.mp4", status=JobStatus.DONE)
        repo.save_job("TedX", tedx_job, drive_url=None, label=None, dedupe_key="d1", duration_s=None)
        repo.save_job("Yixi", yixi_job, drive_url=None, label=None, dedupe_key="d2", duration_s=None)

        assert repo.list_collections() == ["TedX", "Yixi"]

    def test_invalid_collection_name_rejected(self, repo):
        with pytest.raises(ValueError):
            repo.list_videos("bad name with spaces")
        with pytest.raises(ValueError):
            repo.list_videos("")

    def test_missing_uri_raises(self, monkeypatch):
        monkeypatch.delenv("MONGO_URI", raising=False)
        with pytest.raises(ValueError):
            ResultsRepository(uri=None)
