"""Regression tests for lookup, download recovery, and run outcomes.

Run with: python -m unittest discover -s tests -v
No account, network access, or speech model files are needed.
"""

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import main
from src.ai.transcriber import (
    IncompleteAudioError, NoAudioStreamError, Transcriber, _ffmpeg_diagnostic,
)
from src.api.icourse import ICourseClient, VideoLookupError
from src.data.database import Database
from src.pipeline.lecture_runner import LectureRunner
from src.runtime import config
from src.runtime.scheduler import AudioDownloader, AudioHandle, _PendingSpawn


class VideoLookupTests(unittest.TestCase):
    def setUp(self):
        self.client = ICourseClient(Mock())
        self.client.get_sub_info = Mock(return_value={})
        self.client.get_sub_detail = Mock(return_value={})
        self.client.sign_video_url = Mock(side_effect=lambda url, now: url)

    def test_query_string_and_list_sources(self):
        url = "https://media.example/recording.MP4?quality=hd"
        for payload in (
            {"video_list": {"desktop": {"preview_url": url}}},
            {"video_list": [{"preview_url": url}]},
            {"playurl": {"now": 123, "desktop": url}},
            {"playurl": [url]},
            {"content": {"playback": {"url": url}}},
        ):
            with self.subTest(payload=payload):
                self.client.get_sub_info.return_value = payload
                self.assertEqual(self.client.get_video_url("c", "s"), url)

    def test_two_successful_empty_responses_mean_no_video(self):
        self.assertIsNone(self.client.get_video_url("c", "s"))

    def test_lookup_failure_does_not_mean_no_video(self):
        for failing in ("get_sub_info", "get_sub_detail"):
            with self.subTest(endpoint=failing):
                method = getattr(self.client, failing)
                method.side_effect = TimeoutError("temporary outage")
                with self.assertRaises(VideoLookupError):
                    self.client.get_video_url("c", "s")
                method.side_effect = None

    def test_fallback_can_recover_from_failed_primary(self):
        self.client.get_sub_info.side_effect = TimeoutError()
        url = "https://media.example/recording.mp4"
        self.client.get_sub_detail.return_value = {
            "content": {"playback": {"url": url}},
        }
        self.assertEqual(self.client.get_video_url("c", "s"), url)

    def test_refresh_replaces_old_signatures(self):
        client = ICourseClient(Mock())
        client.get_userinfo = Mock(return_value={
            "id": "test-user", "tenant_id": "test-tenant", "phone": "123",
        })
        signed = client.sign_video_url(
            "https://media.example/a.mp4?quality=hd&t=old&clientUUID=old#part",
            now=100,
        )
        query = parse_qs(urlparse(signed).query)
        self.assertEqual(query["quality"], ["hd"])
        self.assertEqual(len(query["t"]), 1)
        self.assertTrue(query["t"][0].startswith("test-user-100-"))
        self.assertNotEqual(query["clientUUID"], ["old"])
        self.assertFalse(urlparse(signed).fragment)

    def test_api_error_with_payload_is_not_silently_treated_as_empty(self):
        client = ICourseClient(Mock())
        client.vpn.get.return_value.json.return_value = {
            "code": 500, "msg": "temporary failure", "data": {"now": 123},
        }
        with self.assertRaises(RuntimeError):
            client.get_sub_info("c", "s")

    def test_pending_release_payload_remains_available(self):
        client = ICourseClient(Mock())
        payload = {"content": {"playback": {"url": "https://media.example/a.mp4"}}}
        client.vpn.get.return_value.json.return_value = {"code": 7001, "data": payload}
        self.assertEqual(client.get_sub_info("c", "s"), payload)


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.downloader = AudioDownloader(directory.name, max_concurrent=1)

    def spawn(self, client):
        token = _PendingSpawn()
        self.downloader._active["s"] = token
        self.downloader._spawn_when_ready(client, "c", "s", token)

    def test_spawn_error_is_propagated_and_slot_released(self):
        client = Mock()
        client.get_video_url.side_effect = TimeoutError("API unavailable")
        self.spawn(client)
        with self.assertRaisesRegex(RuntimeError, "lookup/start failed"):
            self.downloader.get("s")
        self.downloader.release("s")
        self.assertTrue(self.downloader._sem.acquire(blocking=False))
        self.downloader._sem.release()

    def test_confirmed_no_video_returns_none_and_releases_slot(self):
        client = Mock()
        client.get_video_url.return_value = None
        self.spawn(client)
        self.assertIsNone(self.downloader.get("s"))
        self.assertTrue(self.downloader._sem.acquire(blocking=False))
        self.downloader._sem.release()

    def test_cancelled_pending_spawn_does_not_contact_api(self):
        client = Mock()
        self.downloader._spawn_when_ready(client, "c", "s", _PendingSpawn())
        client.get_video_url.assert_not_called()
        self.assertTrue(self.downloader._sem.acquire(blocking=False))
        self.downloader._sem.release()

    def test_stale_lookup_does_not_delete_retry_audio(self):
        path = Path(self.downloader._dir) / "s.raw"
        path.write_bytes(b"new attempt audio")
        current = AudioHandle("s", str(path), Mock(), [])
        client = Mock()
        def finish_old_lookup(*args):
            self.downloader._active["s"] = current
            return "https://media.example/a.mp4"
        client.get_video_url.side_effect = finish_old_lookup
        client.get_stream_params.return_value = ("https://vpn.example/a.mp4", "")
        process = Mock()
        process.stderr = []
        with patch("src.runtime.scheduler.subprocess.Popen", return_value=process):
            self.spawn(client)
        self.assertEqual(path.read_bytes(), b"new attempt audio")
        self.assertIs(self.downloader.get("s"), current)
        process.terminate.assert_called_once()
        self.downloader.release("s")

    def test_ffmpeg_has_connection_recovery_and_timeout(self):
        client = Mock()
        client.get_video_url.return_value = "https://media.example/a.mp4"
        client.get_stream_params.return_value = ("https://vpn.example/a.mp4", "")
        process = Mock()
        process.stderr = []
        with patch("src.runtime.scheduler.subprocess.Popen", return_value=process) as popen:
            self.spawn(client)
        self.downloader.release("s")
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-reconnect_on_network_error") + 1], "1")
        self.assertEqual(command[command.index("-rw_timeout") + 1],
                         str(config.AUDIO_READ_TIMEOUT * 1_000_000))


class RunnerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.conn.close)
        self.db.upsert_course("c", "test course", "teacher")
        self.db.insert_lecture("s", "c", "test lecture", "2026-10-14")
        self.downloader = Mock()
        self.handle = SimpleNamespace(path="unused", process=Mock(), stderr_chunks=[])
        self.downloader.get.return_value = self.handle
        self.scheduler = SimpleNamespace(audio_downloader=self.downloader)
        self.transcriber = Mock()
        self.refresh = Mock()
        self.runner = LectureRunner(Mock(), self.db, self.scheduler,
                                    self.transcriber, Mock(), Mock(),
                                    refresh_session=self.refresh)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(config, "AUDIO_DOWNLOAD_ATTEMPTS", 3))
        stack.enter_context(patch.object(config, "AUDIO_RETRY_DELAY", 0))

    def test_incomplete_download_then_success_saves_only_complete_text(self):
        self.transcriber.transcribe_tail.side_effect = [
            IncompleteAudioError("cut off", 96, 6496, transcript="partial"),
            ("complete transcript", []),
        ]
        result = self.runner._download_transcript("c", "s")
        self.assertEqual(result, ("complete transcript", []))
        self.assertEqual(self.db.get_lecture("s")["transcript"], "complete transcript")
        self.assertEqual(self.downloader.schedule.call_count, 2)
        self.assertEqual(self.downloader.release.call_count, 2)
        self.refresh.assert_called_once()

    def test_connection_failure_is_bounded_and_not_saved_as_no_video(self):
        self.downloader.get.side_effect = RuntimeError("lookup/start failed")
        self.assertEqual(self.runner._download_transcript("c", "s"), (None, None))
        row = self.db.get_lecture("s")
        self.assertEqual(row["error_stage"], "transcribe")
        self.assertEqual(row["error_count"], 1)
        self.assertIsNone(row["transcript"])
        self.assertIsNone(row["processed_at"])
        self.assertEqual(self.downloader.get.call_count, 3)
        self.assertEqual(self.downloader.release.call_count, 3)

    def test_incomplete_downloads_never_persist_partial_text(self):
        self.transcriber.transcribe_tail.side_effect = IncompleteAudioError(
            "cut off", 60, 6471, transcript="partial text",
        )
        self.runner._download_transcript("c", "s")
        row = self.db.get_lecture("s")
        self.assertIsNone(row["transcript"])
        self.assertEqual(row["error_count"], 1)
        self.assertEqual(self.transcriber.transcribe_tail.call_count, 3)

    def test_true_missing_video_has_no_video_status(self):
        self.downloader.get.return_value = None
        self.runner._download_transcript("c", "s")
        self.assertEqual(self.db.get_lecture("s")["error_stage"], "no_video")
        self.transcriber.transcribe_tail.assert_not_called()

    def test_expired_session_is_checked_before_retry(self):
        self.downloader.get.side_effect = [TimeoutError(), self.handle]
        self.transcriber.transcribe_tail.return_value = ("complete", [])
        self.runner._download_transcript("c", "s")
        self.refresh.assert_called_once()
        self.assertEqual(self.db.get_lecture("s")["transcript"], "complete")

    def test_no_audio_is_terminal_and_releases_resources(self):
        self.transcriber.transcribe_tail.side_effect = NoAudioStreamError("no audio")
        self.runner._download_transcript("c", "s")
        self.assertIsNotNone(self.db.get_lecture("s")["processed_at"])
        self.assertEqual(self.downloader.get.call_count, 1)
        self.downloader.release.assert_called_once_with("s")

    def test_empty_transcript_is_not_marked_processed(self):
        self.runner._ppt = Mock()
        self.runner._schedule_next = Mock()
        self.runner._get_transcript = Mock(return_value=("", []))
        self.runner.run("c", "test course", {"sub_id": "s"})
        row = self.db.get_lecture("s")
        self.assertIsNone(row["processed_at"])
        self.assertEqual(row["error_stage"], "transcribe")

    def test_retry_with_real_downloader_fetches_a_fresh_url(self):
        with tempfile.TemporaryDirectory() as directory:
            downloader = AudioDownloader(directory, max_concurrent=1)
            self.runner._scheduler = SimpleNamespace(audio_downloader=downloader)
            client = Mock()
            client.get_video_url.side_effect = [
                "https://media.example/a.mp4?t=first",
                "https://media.example/a.mp4?t=second",
            ]
            client.get_stream_params.side_effect = lambda url: (url, "")
            self.runner._client = client
            process = Mock()
            process.stderr = []
            process.poll.return_value = 0
            self.transcriber.transcribe_tail.side_effect = [
                IncompleteAudioError("cut off", 60, 6471), ("complete", []),
            ]
            with patch("src.runtime.scheduler.subprocess.Popen", return_value=process) as popen:
                result = self.runner._download_transcript("c", "s")
            self.assertEqual(result, ("complete", []))
            self.assertEqual(client.get_video_url.call_count, 2)
            self.assertIn("t=second", popen.call_args.args[0][
                popen.call_args.args[0].index("-i") + 1])
            self.assertIsNone(downloader.get("s"))
            downloader.shutdown()


class DiagnosticTests(unittest.TestCase):
    def test_incomplete_audio_retains_redacted_download_diagnostics(self):
        transcriber = Transcriber()
        def consume(**kwargs):
            transcriber._media_duration = 6496
            transcriber._last_duration = 96
            return "partial", []
        transcriber._consume_pcm_stream = consume
        process = SimpleNamespace(returncode=0, poll=lambda: 0)
        stderr = [b"Connection reset for https://media.example/a.mp4?t=secret\n"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.raw"
            path.touch()
            with self.assertRaises(IncompleteAudioError) as caught:
                transcriber.transcribe_tail(str(path), process, stderr)
        message = str(caught.exception)
        self.assertIn("96s of 6496s", message)
        self.assertIn("Connection reset", message)
        self.assertNotIn("secret", message)
        self.assertNotIn("https://", message)

    def test_headers_are_redacted_before_log_tail(self):
        diagnostic = _ffmpeg_diagnostic(
            b"Cookie: private-session\nAuthorization: Bearer private-token\nConnection timed out"
        )
        self.assertNotIn("private", diagnostic)
        self.assertIn("Connection timed out", diagnostic)

    def test_no_audio_before_file_creation_is_classified(self):
        process = SimpleNamespace(returncode=234, poll=lambda: 234)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(NoAudioStreamError):
                Transcriber().transcribe_tail(
                    str(Path(directory) / "missing.raw"), process,
                    [b"Output file does not contain any stream"],
                )


class RunOutcomeTests(unittest.TestCase):
    def test_silent_failure_is_returned_to_top_level(self):
        db = Mock()
        db.get_lecture.return_value = {"processed_at": None, "error_stage": "no_video"}
        lectures = [("c", "course", {"sub_id": "s"})]
        with patch.object(main, "LectureRunner") as runner, patch.object(main, "_check_session"):
            runner.return_value.run.return_value = None
            failed = main._drive_lectures(Mock(), db, Mock(), Mock(), Mock(),
                                          Mock(), lectures, [])
        self.assertEqual(failed, ["s"])

    def test_failed_run_still_emails_completed_results_and_shuts_down(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(config, "COURSE_IDS", ["c"]))
            for name in ("Database", "Reporter", "Transcriber", "Summarizer", "Emailer",
                         "login_with_retry", "ICourseClient", "_crawl_semester_catalog"):
                mock = stack.enter_context(patch.object(main, name))
                if name == "Database":
                    mock.return_value.sync_dates_from_sub.return_value = 0
            scheduler = stack.enter_context(patch.object(main, "Scheduler"))
            stack.enter_context(patch.object(main, "_enumerate_lectures", return_value=[]))
            stack.enter_context(patch.object(main, "_drive_lectures", return_value=["s"]))
            email = stack.enter_context(patch.object(main, "_send_email"))
            with self.assertRaisesRegex(RuntimeError, "Run incomplete"):
                main.run()
            email.assert_called_once()
            scheduler.return_value.shutdown.assert_called_once()

    def test_no_new_lectures_is_successful(self):
        self.assertEqual(main._drive_lectures(Mock(), Mock(), Mock(), Mock(),
                                             Mock(), Mock(), [], []), [])

    def test_successful_lecture_does_not_hide_a_failed_one(self):
        db = Mock()
        db.get_lecture.side_effect = [
            {"processed_at": "done"},
            {"processed_at": None, "error_stage": "transcribe"},
        ]
        lectures = [("c", "course", {"sub_id": "ok"}),
                    ("c", "course", {"sub_id": "failed"})]
        email_items = []
        with patch.object(main, "LectureRunner") as runner, patch.object(main, "_check_session"):
            runner.return_value.run.side_effect = ["summary", None]
            failed = main._drive_lectures(Mock(), db, Mock(), Mock(), Mock(),
                                          Mock(), lectures, email_items)
        self.assertEqual(failed, ["failed"])
        self.assertEqual([item["sub_id"] for item in email_items], ["ok"])

    def test_course_lookup_failure_is_recorded(self):
        client = Mock()
        client.get_course_detail.side_effect = RuntimeError("API unavailable")
        failed = []
        with patch.object(config, "COURSE_IDS", ["c"]), \
                patch.object(main, "_check_session"), \
                patch.object(main.traceback, "print_exc"):
            lectures = main._enumerate_lectures(client, Mock(), Mock(), failed)
        self.assertEqual(lectures, [])
        self.assertEqual(failed, ["c"])


if __name__ == "__main__":
    unittest.main()
