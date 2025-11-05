"""Backward-compatible import shim for QAController."""

from .system.qa_controller import QAController, QAIntroSpec, SessionFlowCoordinator

__all__ = ["QAController", "QAIntroSpec", "SessionFlowCoordinator"]
