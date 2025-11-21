import { useAuth } from '../utils/authContext';
import { useNavigate } from 'react-router-dom';
import styles from './SelectBuildingPage.module.css';
import Header from '../components/Header';
import Footer from '../components/Footer';

export default function SelectBuildingPage() {
  const { user } = useAuth();
  const schoolName = user?.school_name || '소속 학교';
  const navigate = useNavigate();

  const goEditSelect = () => navigate('/admin/edit-select');

  return (
    <div className={styles.container}>
        <Header/>

        <main className={styles.content}>
        <div className={styles.titleBox}>
            <h2 className={styles.title}>
            이 페이지는<br />
            {schoolName}의<br />
            관리자 페이지에요.
            </h2>

            <h2 className={styles.title}>
            편의시설/배치도를 추가할
            건물을 선택해 주세요
            </h2>
        </div>

        <div className={styles.choiceList}>
            <button
            className={styles.choiceChip}
            onClick={goEditSelect}
            >
            본관
            </button>
        </div>
        </main>

        <Footer/>
    </div>
    );

}
