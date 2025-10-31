import { useAuth } from '../utils/authContext';
import styles from './LogoutButton.module.css';

const LogoutButton = () => {
  const { logout } = useAuth();

  const handleLogout = () => {
    const shouldLogout = window.confirm('정말 로그아웃 하시겠습니까?');

    if (shouldLogout) {
      logout();
    }
  };

  return (
    <button onClick={handleLogout} className={styles.button}>
      {'로그아웃'}
    </button>
  );
};

export default LogoutButton;
