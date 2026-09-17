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


# ── Shards ───────────────────────────────────────────────────────────────

SHARD_URIS = ["mongodb://shard-a/", "mongodb://shard-b/"]


@pytest.fixture
def sharded(monkeypatch):
    """Two independent in-memory clusters. mongomock has no dbStats, so each
    shard's used space is set directly through `usage`."""
    monkeypatch.setattr("core.results_repository.MongoClient", lambda *a, **k: mongomock.MongoClient())
    usage = {"shard-a": 0, "shard-b": 0}
    monkeypatch.setattr("core.results_repository._Shard.used_bytes", lambda self: usage[self.name])
    repo = ResultsRepository(uris=SHARD_URIS, db_name="test_multiarth", shard_limit_mb=1)
    return repo, usage


def _ship(repo, job_id, dedupe_key="d", n_windows=2, collection=COLLECTION):
    job = AnalysisJob(job_id=job_id, video_path=f"/v/{job_id}.mp4", status=JobStatus.DONE)
    return repo.ship_job(
        collection, job, [make_fused(i) for i in range(n_windows)],
        drive_url=None, label=job_id, dedupe_key=dedupe_key, duration_s=10.0,
        artifacts=dict(wordlist={"w": 1}, ngrams=None, collocations=None,
                       spectrogram=None, waveform=None),
    )


def _docs_on(repo, shard_idx, job_id, collection=COLLECTION):
    db = repo.shards[shard_idx].db
    return (
        db[collection + "_videos"].count_documents({"_id": job_id}),
        db[collection + "_fused_windows"].count_documents({"job_id": job_id}),
        db[collection + "_artifacts"].count_documents({"_id": job_id}),
    )


class TestShards:
    def test_env_uris_in_numeric_order(self, monkeypatch):
        from core.results_repository import _uris_from_env
        monkeypatch.setenv("MONGO_URI", "mongodb://one/")
        monkeypatch.setenv("MONGO_URI_10", "mongodb://ten/")
        monkeypatch.setenv("MONGO_URI_2", "mongodb://two/")
        assert _uris_from_env() == ["mongodb://one/", "mongodb://two/", "mongodb://ten/"]

    def test_host_never_includes_credentials(self):
        from core.results_repository import _host
        assert _host("mongodb+srv://user:p%40ss@cluster0.x.mongodb.net/?appName=A") == "cluster0.x.mongodb.net"

    def test_writes_fill_first_shard_while_it_has_room(self, sharded):
        repo, _ = sharded
        assert _ship(repo, "j1") == "shard-a"
        assert _docs_on(repo, 0, "j1") == (1, 2, 1)
        assert _docs_on(repo, 1, "j1") == (0, 0, 0)

    def test_a_full_shard_spills_the_whole_job_to_the_next(self, sharded):
        repo, usage = sharded
        usage["shard-a"] = 1024 * 1024  # at its limit
        assert _ship(repo, "j1") == "shard-b"
        assert _docs_on(repo, 0, "j1") == (0, 0, 0), "job was split across shards"
        assert _docs_on(repo, 1, "j1") == (1, 2, 1)

    def test_job_size_counts_toward_the_limit(self, sharded):
        """A shard with some room left, but not enough for *this* job."""
        repo, usage = sharded
        usage["shard-a"] = int(1024 * 1024 * 0.9) - 2000   # ~2KB left
        assert _ship(repo, "small", n_windows=1) == "shard-a"   # ~0.3KB
        assert _ship(repo, "big", n_windows=20) == "shard-b"    # ~5.4KB

    def test_all_shards_full_raises_instead_of_writing(self, sharded):
        repo, usage = sharded
        usage.update({"shard-a": 10**9, "shard-b": 10**9})
        with pytest.raises(RuntimeError, match="MONGO_URI_3"):
            _ship(repo, "j1")
        assert _docs_on(repo, 0, "j1") == (0, 0, 0)
        assert _docs_on(repo, 1, "j1") == (0, 0, 0)

    def test_retried_ship_stays_on_the_shard_it_started_on(self, sharded):
        """First attempt lands on shard-a; shard-a then reports full. A retry
        must finish the job on shard-a, not start a second copy on shard-b."""
        repo, usage = sharded
        _ship(repo, "j1")
        usage["shard-a"] = 10**9
        assert _ship(repo, "j1") == "shard-a"
        assert _docs_on(repo, 1, "j1") == (0, 0, 0)

    def test_reads_merge_every_shard(self, sharded):
        repo, usage = sharded
        _ship(repo, "old", dedupe_key="d-old")
        _ship(repo, "yixi", dedupe_key="d-y", collection="Yixi")
        usage["shard-a"] = 10**9
        _ship(repo, "new", dedupe_key="d-new")

        assert repo.list_collections() == ["TedX", "Yixi"]
        assert [v["_id"] for v in repo.list_videos(COLLECTION)] == ["old", "new"]

        # a fresh repository (no routing cache) still finds each job
        fresh = ResultsRepository(uris=SHARD_URIS, db_name="test_multiarth", shard_limit_mb=1)
        fresh.shards = repo.shards
        assert len(fresh.get_all_fused(COLLECTION, "new")) == 2
        assert fresh.get_artifacts(COLLECTION, "old")["wordlist"] == {"w": 1}
        assert fresh.get_job(COLLECTION, "new").job_id == "new"
        assert fresh.get_video_doc(COLLECTION, "missing") is None

    def test_dedupe_finds_a_video_on_an_earlier_shard(self, sharded):
        repo, usage = sharded
        _ship(repo, "old", dedupe_key="same-video")
        usage["shard-a"] = 10**9  # new writes now go to shard-b
        assert repo.find_by_dedupe_key(COLLECTION, "same-video") == "old"

    def test_delete_removes_from_every_shard(self, sharded):
        repo, _ = sharded
        _ship(repo, "j1")
        repo.delete_job_data(COLLECTION, "j1")
        assert _docs_on(repo, 0, "j1") == (0, 0, 0)
        assert repo.get_video_doc(COLLECTION, "j1") is None

    def test_unreachable_shard_hides_its_videos_but_not_the_others(self, sharded):
        repo, usage = sharded
        _ship(repo, "a")
        usage["shard-a"] = 10**9
        _ship(repo, "b")

        class Down:
            """Every operation on the unreachable cluster raises."""
            def __getattr__(self, _):
                raise ConnectionError("cluster down")
            def __getitem__(self, _):
                return self

        repo.shards[0].db = Down()
        repo.shards[0].indexed_collections.clear()
        assert repo.list_collections() == ["TedX"]
        assert [v["_id"] for v in repo.list_videos(COLLECTION)] == ["b"]

    def test_reading_does_not_create_collections_on_other_shards(self, sharded):
        repo, _ = sharded
        _ship(repo, "j1", collection="Yixi")
        repo.list_videos("Yixi")
        repo.get_all_fused("Yixi", "j1")
        repo.find_by_dedupe_key("Yixi", "d")
        assert repo.shards[1].db.list_collection_names() == []

    def test_atlas_quota_refusal_moves_the_whole_job_to_the_next_shard(self, sharded, monkeypatch):
        """Our size estimate says shard-a has room, but Atlas disagrees partway
        through the write. The partial job must be removed from shard-a and the
        complete job written to shard-b."""
        from pymongo.errors import OperationFailure
        repo, _ = sharded
        coll = repo.shards[0].db[COLLECTION + "_fused_windows"]

        def over_quota(*a, **k):
            raise OperationFailure("you are over your space quota, using 513 MB of 512 MB", code=8000)
        monkeypatch.setattr(coll, "insert_many", over_quota)

        assert _ship(repo, "j1") == "shard-b"
        assert _docs_on(repo, 0, "j1") == (0, 0, 0), "partial write left behind"
        assert _docs_on(repo, 1, "j1") == (1, 2, 1)
        assert repo.shards[0].full
        assert _ship(repo, "j2") == "shard-b", "a quota-full shard must stay skipped"

    def test_other_write_errors_are_not_treated_as_quota(self, sharded, monkeypatch):
        from pymongo.errors import OperationFailure
        repo, _ = sharded
        coll = repo.shards[0].db[COLLECTION + "_fused_windows"]

        def other(*a, **k):
            raise OperationFailure("not authorized", code=13)
        monkeypatch.setattr(coll, "insert_many", other)
        with pytest.raises(OperationFailure, match="not authorized"):
            _ship(repo, "j1")
        assert not repo.shards[0].full
