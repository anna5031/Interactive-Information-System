import { useEffect } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import AnnotationReviewPage from './AnnotationReviewPage';
import { useFloorPlan } from '../utils/floorPlanContext';
import layoutStyles from './MainPage.module.css';

const AdminReviewPage = () => {
  const navigate = useNavigate();
  const { state, setStage, resetWorkflow } = useFloorPlan();

  const hasResult = Boolean(state.stepOneResult);

  useEffect(() => {
    if (hasResult) {
      setStage('review');
    }
  }, [setStage, hasResult]);

  if (!hasResult) {
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
      <AnnotationReviewPage
        stepOneResult={state.stepOneResult}
        processingResult={state.processingResult}
        onBack={handleBack}
        onFinish={handleFinish}
      />
    </div>
  );
};

export default AdminReviewPage;
