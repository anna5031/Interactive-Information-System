import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { RefreshCw } from 'lucide-react';
import FloorPlanUpload from '../components/floorPlan/FloorPlanUpload';
import StepOneResultCard from '../components/stepOne/StepOneResultCard';
import { uploadFloorPlan } from '../api/floorPlans';
import { getStoredStepOneResults } from '../api/stepOneResults';
import { useFloorPlan } from '../utils/floorPlanContext';
import layoutStyles from './MainPage.module.css';
import styles from './AdminUploadPage.module.css';

const AdminUploadPage = () => {
  const navigate = useNavigate();
  const { state, setStage, setUploadData, loadStepOneResultForEdit } = useFloorPlan();
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [isLoadingResults, setIsLoadingResults] = useState(true);
  const [results, setResults] = useState([]);

  useEffect(() => {
    if (state.skipUploadRedirect) {
      setStage('upload');
      return;
    }

    if (state.stage === 'editor' && state.imageUrl) {
      navigate('/admin/editor', { replace: true });
      return;
    }

    if (state.stage === 'review' && state.stepOneResult) {
      navigate('/admin/review', { replace: true });
      return;
    }

    if (state.stage !== 'upload') {
      setStage('upload');
    }
  }, [state.stage, state.imageUrl, state.stepOneResult, state.skipUploadRedirect, navigate, setStage]);

  const refreshResults = () => {
    setIsLoadingResults(true);
    const stored = getStoredStepOneResults();
    setResults(stored);
    setIsLoadingResults(false);
  };

  useEffect(() => {
    refreshResults();
  }, []);

  const sortedResults = useMemo(() => {
    return [...results].sort((a, b) => {
      const aTime = a?.createdAt ? new Date(a.createdAt).getTime() : 0;
      const bTime = b?.createdAt ? new Date(b.createdAt).getTime() : 0;
      return bTime - aTime;
    });
  }, [results]);

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

  const handleSelectStepTwo = (stepOneResult) => {
    if (!stepOneResult?.id) {
      return;
    }
    navigate(`/admin/step-two/${stepOneResult.id}`);
  };

  const handleEditStepOne = (stepOneResult) => {
    if (!stepOneResult) {
      return;
    }
    loadStepOneResultForEdit(stepOneResult);
    navigate('/admin/editor');
  };

  if (isLoadingResults) {
    return (
      <div className={layoutStyles.centerWrapper}>
        <span>불러오는 중...</span>
      </div>
    );
  }

  if (sortedResults.length === 0) {
    return (
      <div className={styles.emptyWrapper}>
        <FloorPlanUpload onSelectFile={handleSelectFile} isUploading={isUploading} error={error} />
      </div>
    );
  }

  return (
    <div className={layoutStyles.fillContainer}>
      <div className={styles.wrapper}>
        <section className={styles.uploadSection}>
          <FloorPlanUpload onSelectFile={handleSelectFile} isUploading={isUploading} error={error} />
        </section>

        <section className={styles.resultsSection}>
          <div className={styles.resultsHeader}>
            <div>
              <h2 className={styles.resultsTitle}>저장된 도면 ({sortedResults.length})</h2>
              <p className={styles.resultsSubtitle}>다시 검수하거나 2단계 메타데이터 입력을 진행할 도면을 선택하세요.</p>
            </div>
            <button type='button' className={styles.refreshButton} onClick={refreshResults}>
              <RefreshCw size={16} /> 새로고침
            </button>
          </div>

          <div
            className={styles.grid}
            style={{ gridTemplateColumns: sortedResults.length === 1 ? 'repeat(2, minmax(260px, 1fr))' : undefined }}
          >
            {sortedResults.map((result) => (
              <StepOneResultCard
                key={result.id}
                result={result}
                onSelectStepTwo={handleSelectStepTwo}
                onEditStepOne={handleEditStepOne}
              />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
};

export default AdminUploadPage;
