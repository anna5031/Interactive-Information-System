import { Navigate, Route, Routes } from 'react-router-dom';
import { useAuth } from '../utils/authContext';
import { FloorPlanProvider } from '../utils/floorPlanContext';
import LoginPage from '../pages/LoginPage';
import MainPage from '../pages/MainPage';
import LoadingSpinner from '../components/LoadingSpinner';
import AdminUploadPage from '../pages/AdminUploadPage';
import AdminEditorPage from '../pages/AdminEditorPage';
import AdminReviewPage from '../pages/AdminReviewPage';
import AdminStepTwoPage from '../pages/AdminStepTwoPage';
import styles from './AppRouter.module.css';

const AppRouter = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return <div className={styles.container}>{<LoadingSpinner />}</div>;
  }

  if (!user) {
    return (
      <div className={styles.routerWrapper}>
        <Routes>
          <Route path='*' element={<LoginPage />} />
        </Routes>
      </div>
    );
  }

  return (
    <div className={styles.routerWrapper}>
      <FloorPlanProvider>
        <Routes>
          <Route path='/admin' element={<MainPage />}>
            <Route index element={<Navigate to='upload' replace />} />
            <Route path='upload' element={<AdminUploadPage />} />
            <Route path='editor' element={<AdminEditorPage />} />
            <Route path='review' element={<AdminReviewPage />} />
            <Route path='step-two/:stepOneId' element={<AdminStepTwoPage />} />
            <Route path='*' element={<Navigate to='upload' replace />} />
          </Route>
          <Route path='*' element={<Navigate to='/admin/upload' replace />} />
        </Routes>
      </FloorPlanProvider>
    </div>
  );
};

export default AppRouter;
