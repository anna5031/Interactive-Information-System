import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import FloorPlanUpload from '../components/floorPlan/FloorPlanUpload';
import { uploadFloorPlan } from '../api/floorPlans';
import { useFloorPlan } from '../utils/floorPlanContext';
import layoutStyles from './MainPage.module.css';

const AdminUploadPage = () => {
  const navigate = useNavigate();
  const { state, setStage, setUploadData } = useFloorPlan();
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (state.skipUploadRedirect) {
      setStage('upload');
      return;
    }

    if (state.stage === 'editor' && state.imageUrl) {
      navigate('/admin/editor', { replace: true });
      return;
    }

    if (state.stage === 'review' && (state.savedYoloText || state.savedWallText || state.savedDoorText)) {
      navigate('/admin/review', { replace: true });
      return;
    }

    if (state.stage !== 'upload') {
      setStage('upload');
    }
  }, [
    state.stage,
    state.imageUrl,
    state.savedYoloText,
    state.savedWallText,
    state.savedDoorText,
    state.skipUploadRedirect,
    navigate,
    setStage
  ]);

  const handleSelectFile = async (file) => {
    setError(null);
    setIsUploading(true);
    try {
      const result = await uploadFloorPlan(file);
      setUploadData(result);
      navigate('/admin/editor');
    } catch (uploadError) {
      console.error(uploadError);
      setError('업로드에 실패했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className={layoutStyles.centerWrapper}>
      <FloorPlanUpload onSelectFile={handleSelectFile} isUploading={isUploading} error={error} />
    </div>
  );
};

export default AdminUploadPage;
