import { useNavigate } from 'react-router-dom';
import LocalLoginButton from '../components/LocalLoginButton';
import styles from './LoginPage.module.css';

const LoginPage = () => {
  const navigate = useNavigate();

  return (
    <div className={styles.container}>
      <div className={styles.loginBox}>
        <LocalLoginButton onSuccess={() => navigate('/admin/upload')} />
      </div>
    </div>
  );
};

export default LoginPage;
