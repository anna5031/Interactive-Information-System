import { useNavigate } from 'react-router-dom';
import { useAuth } from '../utils/authContext';
import styles from './EditSelectionPage.module.css';
import Header from '../components/Header';
import Footer from '../components/Footer';

export default function EditSelectionPage() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const schoolName = user?.school_name || '소속 학교';

  return (
    <div className={styles.container}>
      <Header />

      <main className={styles.body}>
        {/* load school name from API (if logged in) */}
        
        <section className={styles.leftColumn}>

          <div className={styles.onboardingText}>
            <h2>이 페이지는<br/>{schoolName}의 관리자 페이지에요.</h2>
            <h2 style={{marginTop:16}}>편집할 콘텐츠를 선택해 주세요</h2>
          </div>

          <div className={styles.choiceList}>
            <button className={styles.choiceChip} onClick={() => navigate('/admin/upload')}>
              설계도 추가하기
            </button>
            <button className={styles.choiceChip} onClick={() => navigate('/admin/upload-facility')}>
              편의시설 사진 추가하기
            </button>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
