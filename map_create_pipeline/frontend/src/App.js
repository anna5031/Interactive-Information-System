import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from './utils/authContext';
import AppRouter from './router/AppRouter';
import styles from './App.module.css';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <div className={styles.app}>
          <AppRouter />
        </div>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
