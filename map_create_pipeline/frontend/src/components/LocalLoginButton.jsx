import { useState } from 'react';
import { useAuth } from '../utils/authContext';
import styles from './LocalLoginButton.module.css';

const LocalLoginButton = () => {
  const [userId, setUserId] = useState('');
  const { login } = useAuth();

  const handleLocalLogin = async () => {
    if (userId.trim()) {
      await login(userId, userId, `${userId}@local.com`);
    }
  };

  return (
    <div className={styles.container}>
      <input
        type='text'
        placeholder='Enter User ID'
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
        className={styles.input}
        onKeyDown={(e) => e.key === 'Enter' && handleLocalLogin()}
      />
      <button onClick={handleLocalLogin} className={styles.button}>
        Login
      </button>
    </div>
  );
};

export default LocalLoginButton;
