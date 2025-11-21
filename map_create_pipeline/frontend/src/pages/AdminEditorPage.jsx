import { useEffect, useState } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import FloorPlanEditorPage from './FloorPlanEditorPage';
import { saveAnnotations } from '../api/floorPlans';
import { useFloorPlan } from '../utils/floorPlanContext';
import layoutStyles from './MainPage.module.css';
import styles from './AdminEditorPage.module.css';

const AdminEditorPage = () => {
  const navigate = useNavigate();
  const {
    state,
    setStage,
    updateBoxes,
    updateLines,
    updatePoints,
    resetWorkflow,
    wallFilter,
    setWallFilter,
    wallBaseLines,
    updateWallBaseLines,
    setCalibrationLine,
    setCalibrationLengthMeters,
  } = useFloorPlan();
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (state.imageUrl && state.stage !== 'editor') {
      setStage('editor');
    }
  }, [setStage, state.imageUrl, state.stage]);

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

  const handleSubmit = async ({ boxes, lines, points, baseLines, wallFilterState, freeSpacePreview }) => {
    if (!Number.isFinite(state.metersPerPixel) || state.metersPerPixel <= 0) {
      setError('축척 정보가 없어 저장할 수 없습니다. 기준선 길이를 입력한 뒤 다시 시도해 주세요.');
      return;
    }
    setIsSaving(true);
    setError(null);
    try {
      updateBoxes(boxes);
      updateLines(lines);
      updatePoints(points);
      if (Array.isArray(baseLines)) {
        updateWallBaseLines(baseLines);
      }
      if (wallFilterState) {
        setWallFilter(wallFilterState);
      }
      const existingRequestId =
        state.processingResult?.request_id ??
        state.processingResult?.requestId ??
        state.stepOneResult?.processingResult?.request_id ??
        state.stepOneResult?.processingResult?.requestId ??
        null;
      const saveResponse = await saveAnnotations({
        fileName: state.fileName,
        floorLabel: state.floorLabel,
        floorValue: state.floorValue,
        calibrationLine: state.calibrationLine,
        calibrationLengthMeters: state.calibrationLengthMeters,
        imageUrl: state.imageUrl,
        boxes,
        lines,
        points,
        baseLines,
        wallFilter: wallFilterState ?? wallFilter,
        rawObjectDetectText: state.rawObjectDetectText,
        rawWallText: state.rawWallText,
        rawDoorText: state.rawDoorText,
        imageWidth: state.imageWidth,
        imageHeight: state.imageHeight,
        sourceOriginalId: state.stepOneOriginalId,
        sourceImagePath: state.fileName,
        requestId: existingRequestId,
        freeSpacePreview,
      });
      if (saveResponse?.anchorLineRecoveryCount > 0 && typeof window !== 'undefined') {
        window.alert(
          `문 포인트가 붙어 있는 벽 선 ${saveResponse.anchorLineRecoveryCount}개가 자동으로 저장본에 포함되었습니다. 필요 시 필터 설정을 조정해 주세요.`
        );
      }
      resetWorkflow({ skipUploadRedirect: true });
      setStage('upload');
      navigate('/admin/upload');
    } catch (saveError) {
      console.error(saveError);
      setError(saveError?.message || '저장 중 오류가 발생했습니다.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className={`${layoutStyles.fillContainer} ${styles.editorLayout}`}>
      <FloorPlanEditorPage
        fileName={state.fileName || 'floor-plan.png'}
        imageUrl={state.imageUrl}
        initialBoxes={state.boxes}
        initialLines={state.lines}
        initialWallBaseLines={wallBaseLines ?? state.lines}
        initialPoints={state.points}
        imageWidth={state.imageWidth}
        imageHeight={state.imageHeight}
        onBaseLinesChange={updateWallBaseLines}
        onCancel={handleCancel}
        onSubmit={handleSubmit}
        isSaving={isSaving}
        onBoxesChange={updateBoxes}
        onLinesChange={updateLines}
        onPointsChange={updatePoints}
        wallFilter={wallFilter}
        onWallFilterChange={setWallFilter}
        errorMessage={error}
        calibrationLine={state.calibrationLine}
        calibrationLengthMeters={state.calibrationLengthMeters}
        metersPerPixel={state.metersPerPixel}
        onCalibrationLineChange={setCalibrationLine}
        onCalibrationLengthChange={setCalibrationLengthMeters}
      />
    </div>
  );
};

export default AdminEditorPage;
