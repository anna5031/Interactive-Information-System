import { useEffect, useState } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import FloorPlanEditorPage from './FloorPlanEditorPage';
import { saveAnnotations } from '../api/floorPlans';
import { useFloorPlan } from '../utils/floorPlanContext';
import layoutStyles from './MainPage.module.css';

const AdminEditorPage = () => {
  const navigate = useNavigate();
  const {
    state,
    setStage,
    updateBoxes,
    updateLines,
    updatePoints,
    setStepOneResult,
    setProcessingResult,
    resetWorkflow,
  } = useFloorPlan();
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (state.imageUrl) {
      setStage('editor');
    }
  }, [setStage, state.imageUrl]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return () => {};
    }

    const handlePopState = () => {
      const nextPath = window.location.pathname;
      if (nextPath.startsWith('/admin/upload')) {
        resetWorkflow({ skipUploadRedirect: true });
      } else if (nextPath.startsWith('/admin/review')) {
        setStage('review');
      } else if (nextPath.startsWith('/admin/editor')) {
        setStage('editor');
      }
    };

    window.addEventListener('popstate', handlePopState);

    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, [setStage, resetWorkflow]);

  if (!state.imageUrl) {
    return <Navigate to='/admin/upload' replace />;
  }

  const handleCancel = () => {
    resetWorkflow({ skipUploadRedirect: true });
    navigate('/admin/upload', { replace: true });
  };

  const handleSubmit = async ({ boxes, lines, points }) => {
    setIsSaving(true);
    setError(null);
    try {
      updateBoxes(boxes);
      updateLines(lines);
      updatePoints(points);
      const { stepOneResult, processingResult } = await saveAnnotations({
        fileName: state.fileName,
        imageUrl: state.imageUrl,
        boxes,
        lines,
        points,
        rawYoloText: state.rawYoloText,
        rawWallText: state.rawWallText,
        rawDoorText: state.rawDoorText,
        imageWidth: state.imageWidth,
        imageHeight: state.imageHeight,
        sourceOriginalId: state.stepOneOriginalId,
        sourceImagePath: state.fileName,
      });
      setStepOneResult(stepOneResult, { processingResult });
      setProcessingResult(processingResult ?? null);
      setStage('review');
      navigate('/admin/review');
    } catch (saveError) {
      console.error(saveError);
      setError('저장 중 오류가 발생했습니다.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className={layoutStyles.fillContainer}>
      <FloorPlanEditorPage
        fileName={state.fileName || 'floor-plan.png'}
        imageUrl={state.imageUrl}
        initialBoxes={state.boxes}
        initialLines={state.lines}
        initialPoints={state.points}
        onCancel={handleCancel}
        onSubmit={handleSubmit}
        isSaving={isSaving}
        onBoxesChange={updateBoxes}
        onLinesChange={updateLines}
        onPointsChange={updatePoints}
        errorMessage={error}
      />
    </div>
  );
};

export default AdminEditorPage;
