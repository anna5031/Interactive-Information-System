import LocalLoginButton from '../components/LocalLoginButton';
import styles from './LoginPage.module.css';

const LoginPage = () => {
  return (
    <div className={styles.container}>
      <div className={styles.loginBox}>
        <LocalLoginButton />
      </div>
    </div>
  );
};

export default LoginPage;
