import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import appConfig, { resolveRouteForAction, resolveWebSocketUrl } from '../config/appConfig';

const COMMAND_ACK_RECEIVED = 'received';
const COMMAND_ACK_COMPLETED = 'completed';

const CONNECTION_STATUS = {
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
};

function createInitialQaState() {
  return {
    status: 'idle',
    initialPrompt: null,
    speakingMessage: null,
    displayMessage: '안내가 곧 시작됩니다.',
    lastAction: null,
    lastContext: null,
  };
}

export default function useWebSocketController({ onNavigate }) {
  const [connectionStatus, setConnectionStatus] = useState(CONNECTION_STATUS.CONNECTING);
  const [latestHomography, setLatestHomography] = useState(null);
  const [currentScreenCommand, setCurrentScreenCommand] = useState(null);
  const [commandForNavigation, setCommandForNavigation] = useState(null);
  const [qaState, setQaState] = useState(() => createInitialQaState());

  const socketRef = useRef(null);
  const shouldReconnectRef = useRef(true);
  const reconnectTimerRef = useRef(null);
  const disconnectGraceTimerRef = useRef(null);
  const ackStatusRef = useRef(new Map());
  const activeCommandRef = useRef(null);
  const onNavigateRef = useRef(onNavigate);
  const connectRef = useRef(null);

  useEffect(() => {
    onNavigateRef.current = onNavigate;
  }, [onNavigate]);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const clearDisconnectGraceTimer = useCallback(() => {
    if (disconnectGraceTimerRef.current) {
      clearTimeout(disconnectGraceTimerRef.current);
      disconnectGraceTimerRef.current = null;
    }
  }, []);

  const sendAck = useCallback((command, status) => {
    if (!command) {
      return;
    }

    const payload = {
      type: 'ack',
      action: command.action,
      status,
      commandId: command.id,
      sequence: command.sequence,
      timestamp: Date.now(),
    };

    ackStatusRef.current.set(command.id, status);

    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }

    try {
      socket.send(JSON.stringify(payload));
    } catch (error) {
      // Swallow send errors; reconnection logic will retry automatically.
      console.error('[WebSocket] Failed to send ACK', error);
    }
  }, []);

  const completeCommand = useCallback(
    (command) => {
      if (!command) {
        return;
      }

      sendAck(command, COMMAND_ACK_COMPLETED);
      if (activeCommandRef.current?.id === command.id) {
        activeCommandRef.current = null;
      }
      setCommandForNavigation(null);
    },
    [sendAck],
  );

  const handleCommandMessage = useCallback(
    (message) => {
      const { action } = message;
      if (!action) {
        return;
      }

      const commandId =
        message.commandId ??
        message.id ??
        message.sequence ??
        `${action}-${message.timestamp ?? Date.now()}`;

      const existingStatus = ackStatusRef.current.get(commandId);
      const command = {
        id: commandId,
        action,
        context: message.context ?? {},
        sequence: message.sequence ?? null,
        raw: message,
        route: resolveRouteForAction(action),
      };

      switch (action) {
        case 'start_qa':
          setQaState(() => ({
            status: 'prompt',
            initialPrompt: message.context?.initialPrompt ?? '',
            speakingMessage: null,
            displayMessage: message.context?.initialPrompt ?? '안내가 곧 시작됩니다.',
            lastAction: action,
            lastContext: message.context ?? {},
          }));
          break;
        case 'start_listening':
          setQaState((prev) => ({
            ...prev,
            status: 'listening',
            speakingMessage: null,
            displayMessage: message.context?.message ?? '듣는 중...',
            lastAction: action,
            lastContext: message.context ?? {},
          }));
          break;
        case 'stop_listening':
          setQaState((prev) => ({
            ...prev,
            status: 'thinking',
            displayMessage: prev.displayMessage,
            lastAction: action,
            lastContext: message.context ?? {},
          }));
          break;
        case 'start_thinking':
          setQaState((prev) => ({
            ...prev,
            status: 'thinking',
            speakingMessage: null,
            displayMessage: message.context?.message ?? '생각 중...',
            lastAction: action,
            lastContext: message.context ?? {},
          }));
          break;
        case 'start_speaking':
          setQaState((prev) => ({
            ...prev,
            status: 'speaking',
            speakingMessage: message.context?.message ?? '',
            displayMessage: message.context?.message ?? prev.displayMessage,
            lastAction: action,
            lastContext: message.context ?? {},
          }));
          break;
        case 'stop_speaking':
          setQaState((prev) => ({
            ...prev,
            status: 'awaiting_listening',
            speakingMessage: null,
            displayMessage: prev.displayMessage,
            lastAction: action,
            lastContext: message.context ?? {},
          }));
          break;
        case 'start_nudge':
        case 'start_guidance':
        case 'start_landing':
        case 'stop_all':
        case 'enter_idle':
        case 'start_idle':
          setQaState(() => createInitialQaState());
          break;
        default:
          break;
      }

      // Already completed – confirm completion and ignore.
      if (existingStatus === COMMAND_ACK_COMPLETED) {
        sendAck(command, COMMAND_ACK_COMPLETED);
        return;
      }

      // Already processing – refresh "received" ACK and skip.
      if (activeCommandRef.current?.id === commandId && existingStatus === COMMAND_ACK_RECEIVED) {
        sendAck(command, COMMAND_ACK_RECEIVED);
        return;
      }

      activeCommandRef.current = command;
      setCurrentScreenCommand({ action: command.action, context: command.context, commandId });
      setCommandForNavigation(command);
      sendAck(command, COMMAND_ACK_RECEIVED);
    },
    [sendAck],
  );

  const handleMessage = useCallback(
    (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch (error) {
        console.error('[WebSocket] Failed to parse message', error);
        return;
      }

      switch (payload.type) {
        case 'homography':
          setLatestHomography({
            matrix: payload.matrix,
            timestamp: payload.timestamp ?? Date.now(),
            raw: payload,
          });
          break;
        case 'command':
          handleCommandMessage(payload);
          break;
        default:
          break;
      }
    },
    [handleCommandMessage],
  );

  const scheduleReconnect = useCallback(() => {
    if (!shouldReconnectRef.current || reconnectTimerRef.current) {
      return;
    }

    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null;
      connectRef.current?.();
    }, appConfig.websocket.reconnectDelayMs);
  }, []);

  const handleDisconnect = useCallback(() => {
    setConnectionStatus(CONNECTION_STATUS.DISCONNECTED);
    if (!disconnectGraceTimerRef.current) {
      disconnectGraceTimerRef.current = setTimeout(() => {
        disconnectGraceTimerRef.current = null;
        const fallbackRoute = resolveRouteForAction('enter_idle');
        onNavigateRef.current?.(fallbackRoute);
      }, appConfig.websocket.disconnectGraceMs);
    }
    scheduleReconnect();
  }, [scheduleReconnect]);

  const connect = useCallback(() => {
    clearReconnectTimer();

    const url = resolveWebSocketUrl();
    const socket = new WebSocket(url);
    socketRef.current = socket;
    setConnectionStatus(CONNECTION_STATUS.CONNECTING);

    socket.onopen = () => {
      ackStatusRef.current.clear();
      clearDisconnectGraceTimer();
      setConnectionStatus(CONNECTION_STATUS.CONNECTED);
    };

    socket.onmessage = handleMessage;

    socket.onerror = () => {
      socket.close();
    };

    socket.onclose = () => {
      handleDisconnect();
    };
  }, [clearDisconnectGraceTimer, clearReconnectTimer, handleDisconnect, handleMessage]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    const startTimer = setTimeout(() => {
      connectRef.current?.();
    }, 0);

    return () => {
      shouldReconnectRef.current = false;
      clearTimeout(startTimer);
      clearReconnectTimer();
      clearDisconnectGraceTimer();
      const socket = socketRef.current;
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.close();
      }
      socketRef.current = null;
    };
  }, [clearDisconnectGraceTimer, clearReconnectTimer, connect]);

  useEffect(() => {
    if (!commandForNavigation) {
      return;
    }

    const completionTimer = setTimeout(() => {
      if (commandForNavigation.route) {
        onNavigateRef.current?.(commandForNavigation.route);
      }
      completeCommand(commandForNavigation);
    }, 0);

    return () => {
      clearTimeout(completionTimer);
    };
  }, [commandForNavigation, completeCommand]);

  const state = useMemo(
    () => ({
      connectionStatus,
      latestHomography,
      currentScreenCommand,
      qaState,
    }),
    [connectionStatus, currentScreenCommand, latestHomography, qaState],
  );

  return state;
}
