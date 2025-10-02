import { useEffect } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import AnnotationReviewPage from './AnnotationReviewPage';
import { useFloorPlan } from '../utils/floorPlanContext';
import layoutStyles from './MainPage.module.css';

const AdminReviewPage = () => {
  const navigate = useNavigate();
  const { state, setStage, resetWorkflow } = useFloorPlan();

  useEffect(() => {
    if (state.savedText) {
      setStage('review');
    }
  }, [setStage, state.savedText]);

  if (!state.savedText) {
    if (state.imageUrl) {
      return <Navigate to='/admin/editor' replace />;
    }
    return <Navigate to='/admin/upload' replace />;
  }

  const handleBack = () => {
    setStage('editor');
    navigate('/admin/editor');
  };

  const handleFinish = () => {
    resetWorkflow();
    navigate('/admin/upload');
  };

  return (
    <div className={layoutStyles.fillContainer}>
      <AnnotationReviewPage savedText={state.savedText} onBack={handleBack} onFinish={handleFinish} />
    </div>
  );
};

export default AdminReviewPage;
