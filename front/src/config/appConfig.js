const DEFAULT_WS_PORT = 8765;

const appConfig = {
  websocket: {
    host: null, // null이면 window.location.hostname 사용
    port: DEFAULT_WS_PORT,
    reconnectDelayMs: 2000,
    disconnectGraceMs: 10000,
  },
  commands: {
    actionRouteMap: {
      start_nudge: '/nudge',
      start_guidance: '/guidance',
      enter_idle: '/',
      start_idle: '/',
      start_landing: '/',
      start_qa: '/qa',
      start_listening: '/qa',
      stop_listening: '/qa',
      start_thinking: '/qa',
      start_speaking: '/qa',
      stop_speaking: '/qa',
      show_overlay: '/qa',
      stop_all: '/',
    },
  },
};

export function resolveWebSocketUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const hostname = appConfig.websocket.host ?? window.location.hostname ?? 'localhost';
  const port = appConfig.websocket.port;
  return `${protocol}://${hostname}:${port}`;
}

export function resolveRouteForAction(action) {
  const { actionRouteMap } = appConfig.commands;
  return actionRouteMap[action] ?? '/';
}

export default appConfig;
