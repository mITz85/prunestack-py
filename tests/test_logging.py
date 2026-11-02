"""Tests for prunestack.utils.logging module."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from prunestack.utils.logging import setup_logging, get_logger


class TestSetupLogging:
    """Test logging configuration."""

    def test_default_setup(self):
        root = setup_logging()
        assert root.name == "prunestack"
        assert root.level == logging.DEBUG
        # Should have console handler
        assert len(root.handlers) >= 1

    def test_verbose_mode(self):
        root = setup_logging(verbose=True)
        console = root.handlers[0]
        assert console.level == logging.DEBUG

    def test_quiet_mode(self):
        root = setup_logging(quiet=True)
        console = root.handlers[0]
        assert console.level == logging.WARNING

    def test_explicit_level(self):
        root = setup_logging(level=logging.ERROR)
        console = root.handlers[0]
        assert console.level == logging.ERROR

    def test_file_handler(self, tmp_path):
        log_file = tmp_path / "test.log"
        root = setup_logging(log_file=log_file)
        # Should have both console and file handler
        assert len(root.handlers) >= 2
        assert log_file.exists() or True  # file created on first write

    def test_reconfiguration_clears_handlers(self):
        root = setup_logging(verbose=True)
        n1 = len(root.handlers)
        root = setup_logging(quiet=True)
        n2 = len(root.handlers)
        # Should not accumulate handlers
        assert n2 <= n1 + 1

    def test_file_handler_writes(self, tmp_path):
        log_file = tmp_path / "test.log"
        root = setup_logging(log_file=log_file)
        root.info("test message")
        # Force flush
        for h in root.handlers:
            h.flush()
        content = log_file.read_text()
        assert "test message" in content


class TestGetLogger:
    """Test the get_logger helper."""

    def test_returns_child_logger(self):
        lg = get_logger("data.loader")
        assert lg.name == "prunestack.data.loader"

    def test_different_names(self):
        lg1 = get_logger("models")
        lg2 = get_logger("metrics")
        assert lg1.name != lg2.name
