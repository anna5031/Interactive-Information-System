import { MemoryRouter, Route, Routes, useNavigate } from 'react-router-dom';
import { useMemo } from 'react';
import styles from './App.module.css';
import LandingScreen from './components/screens/LandingScreen';
import NudgeScreen from './components/screens/NudgeScreen';
import GuidanceScreen from './components/screens/GuidanceScreen';
import QaScreen from './components/screens/QaScreen';
import { AppStateProvider } from './state/AppStateContext';
import useWebSocketController from './hooks/useWebSocketController';

function App() {
  return (
    <div className={styles.app}>
      <MemoryRouter>
        <AppRouter />
      </MemoryRouter>
    </div>
  );
}

function AppRouter() {
  const navigate = useNavigate();
  const controllerState = useWebSocketController({
    onNavigate: navigate,
  });

  const stateValue = useMemo(() => controllerState, [controllerState]);

  return (
    <AppStateProvider value={stateValue}>
      <Routes>
        <Route path="/" element={<LandingScreen />} />
        <Route path="/nudge" element={<NudgeScreen />} />
        <Route path="/guidance" element={<GuidanceScreen />} />
        <Route path="/qa" element={<QaScreen />} />
        <Route path="*" element={<LandingScreen />} />
      </Routes>
    </AppStateProvider>
  );
}

export default App;
