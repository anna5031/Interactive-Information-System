import { createContext, useContext } from 'react';
import PropTypes from 'prop-types';

const AppStateContext = createContext(null);

export function AppStateProvider({ value, children }) {
  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const context = useContext(AppStateContext);

  if (!context) {
    throw new Error('useAppState must be used within an AppStateProvider');
  }

  return context;
}

export default AppStateContext;

AppStateProvider.propTypes = {
  value: PropTypes.shape({
    connectionStatus: PropTypes.string.isRequired,
    latestHomography: PropTypes.object,
    currentScreenCommand: PropTypes.object,
    detectionState: PropTypes.shape({
      status: PropTypes.string.isRequired,
      lastCommandId: PropTypes.string,
    }).isRequired,
    qaState: PropTypes.shape({
      status: PropTypes.string.isRequired,
      initialPrompt: PropTypes.string,
      speakingMessage: PropTypes.string,
      lastAction: PropTypes.string,
      lastContext: PropTypes.object,
    }).isRequired,
  }).isRequired,
  children: PropTypes.node.isRequired,
};
