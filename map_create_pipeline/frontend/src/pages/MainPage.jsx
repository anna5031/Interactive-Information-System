import { Outlet, useLocation } from 'react-router-dom';
import LogoutButton from '../components/LogoutButton';
import styles from './MainPage.module.css';

function MainPage() {
  const location = useLocation();
  const showLogout = location.pathname === '/admin/upload' || location.pathname === '/admin';
  const isEditorRoute = location.pathname.startsWith('/admin/editor');

  return (
    <div className={`${styles.page} ${isEditorRoute ? styles.editorPage : ''}`}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>관리자 페이지</h1>
          <p className={styles.subtitle}>도면 객체를 검수하고 수정하세요.</p>
        </div>
        {showLogout && <LogoutButton />}
      </header>

      <main className={`${styles.content} ${isEditorRoute ? styles.editorContent : ''}`}>
        <Outlet />
      </main>
    </div>
  );
}

export default MainPage;
