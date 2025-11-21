import { Navigate, Route, Routes } from 'react-router-dom';
import { useAuth } from '../utils/authContext';
import { FloorPlanProvider } from '../utils/floorPlanContext';
import LoginPage from '../pages/LoginPage';
import MainPage from '../pages/MainPage';
import LoadingSpinner from '../components/LoadingSpinner';
import AdminUploadPage from '../pages/AdminUploadPage';
import AdminEditorPage from '../pages/AdminEditorPage';
import AdminReviewPage from '../pages/AdminReviewPage';
import AdminGraphEditorPage from '../pages/AdminGraphEditorPage';
import AdminStepThreePage from '../pages/AdminStepThreePage';
import SelectBuildingPage from '../pages/SelectBuildingPage';
import EditSelectionPage from '../pages/EditSelectionPage';
import FacilityUploadPage from '../pages/FacilityUploadPage';
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
          <Route path='/' element={<LoginPage />} />
          <Route path='*' element={<Navigate to='/' replace />} />
        </Routes>
      </div>
    );
  }

  return (
    <div className={styles.routerWrapper}>
      <FloorPlanProvider>
        <Routes>
          <Route path='/admin/select' element={<SelectBuildingPage />} />
          <Route path='/admin/edit-select' element={<EditSelectionPage />} />
          <Route path='/admin/upload-facility' element={<FacilityUploadPage />} />

          <Route path='/admin' element={<MainPage />}>
            <Route index element={<Navigate to='upload' replace />} />
            <Route path='upload' element={<AdminUploadPage />} />
            <Route path='editor' element={<AdminEditorPage />} />
            <Route path='review' element={<AdminReviewPage />} />
            <Route path='graph/:requestId' element={<AdminGraphEditorPage />} />
            <Route path='step-three/:stepOneId' element={<AdminStepThreePage />} />
            <Route path='*' element={<Navigate to='upload' replace />} />
          </Route>
          <Route path='*' element={<Navigate to='/admin/select' replace />} />
        </Routes>
      </FloorPlanProvider>
    </div>
  );
};

export default AppRouter;
