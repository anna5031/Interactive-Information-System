from __future__ import annotations

from dataclasses import dataclass

from devices.manager import DeviceManager
from session.runner import SessionRunnerFactory, SessionRunner
from websocket.connection import ClientConnection


@dataclass
class Application:
    device_manager: DeviceManager
    session_factory: SessionRunnerFactory

    def startup(self) -> None:
        self.device_manager.initialize()

    def shutdown(self) -> None:
        self.session_factory.shutdown()
        self.device_manager.shutdown()

    def create_session_runner(self, connection: ClientConnection) -> SessionRunner:
        return self.session_factory.create(connection)
