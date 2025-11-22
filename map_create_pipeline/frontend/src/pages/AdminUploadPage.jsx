import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { RefreshCw } from 'lucide-react';
import FloorPlanUpload from '../components/floorPlan/FloorPlanUpload';
import StepOneResultCard from '../components/stepOne/StepOneResultCard';
import {
  uploadFloorPlan,
  fetchStoredFloorPlanSummaries,
  processStepOneForStepTwo,
  deleteStoredFloorPlan,
  fetchFloorPlanFloors,
} from '../api/floorPlans';
import { fetchStepThreeStatuses } from '../api/stepThreeResults';
import { deleteStepOneResult, getStoredStepOneResults } from '../api/stepOneResults';
import { useFloorPlan } from '../utils/floorPlanContext';
import { buildStepOneRecordFromStoredResult } from '../utils/processingResult';
import layoutStyles from './MainPage.module.css';
import styles from './AdminUploadPage.module.css';

const normalizeFloorLabel = (value) => (value || '').trim();

const normalizeFloorValue = (value) => (value || '').trim().toUpperCase();

const recordFloorLabel = (record) => normalizeFloorLabel(record?.metadata?.floorLabel ?? record?.floorLabel ?? '');

const recordFloorValue = (record) => normalizeFloorValue(record?.metadata?.floorValue ?? record?.floorValue ?? '');

const hasDuplicateFloor = (records, floorLabel, floorValue) => {
  const normalizedLabel = normalizeFloorLabel(floorLabel);
  const normalizedValue = normalizeFloorValue(floorValue);
  if (!normalizedLabel && !normalizedValue) {
    return false;
  }
  return (records || []).some((record) => {
    const candidateLabel = recordFloorLabel(record);
    const candidateValue = recordFloorValue(record);
    return (
      (normalizedValue && candidateValue && candidateValue === normalizedValue) ||
      (normalizedLabel && candidateLabel && candidateLabel === normalizedLabel)
    );
  });
};

const AdminUploadPage = () => {
  const navigate = useNavigate();
  const { state, setStage, setUploadData, loadStepOneResultForEdit } = useFloorPlan();
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [isLoadingResults, setIsLoadingResults] = useState(true);
  const [results, setResults] = useState([]);
  const [stepThreeStatuses, setStepThreeStatuses] = useState({});
  const [graphOpeningStepOneId, setGraphOpeningStepOneId] = useState(null);
  const [deletingRequestId, setDeletingRequestId] = useState(null);

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

  const refreshResults = async ({ showLoader = true, requireRemote = false } = {}) => {
    if (showLoader) {
      setIsLoadingResults(true);
    }
    try {
      // 1. 로컬 스토리지에 있는 데이터 가져오기 (업로드 전인 임시 저장본 등)
      const localResults = getStoredStepOneResults();

      // 2. 서버에서 최신 목록 가져오기
      let serverSummaries = [];
      let summariesError = null;
      try {
        serverSummaries = await fetchStoredFloorPlanSummaries();
      } catch (summaryError) {
        summariesError = summaryError;
        console.error('Failed to load stored floor plan summaries', summaryError);
      }

      if (summariesError && requireRemote) {
        throw summariesError;
      }

      // 3. 서버 데이터를 애플리케이션 레코드 형태로 변환 (로컬 저장 X)
      const serverRecords = [];
      if (Array.isArray(serverSummaries)) {
        for (const summary of serverSummaries) {
          try {
            const record = buildStepOneRecordFromStoredResult(summary);
            if (record) {
              // 서버 출처임을 명시
              serverRecords.push({ ...record, origin: 'server' });
            }
          } catch (hydrateError) {
            console.error('Failed to hydrate stored floor plan result', hydrateError);
          }
        }
      }

      // 4. 병합 로직: 서버 데이터 우선 + 서버에 없는 로컬 데이터 추가
      // 서버 데이터의 ID(request_id 또는 id)를 Set으로 관리하여 중복 배제
      const serverIds = new Set();
      serverRecords.forEach((item) => {
        if (item.requestId) serverIds.add(item.requestId);
        if (item.id) serverIds.add(item.id);
        // processingResult 내부의 request_id도 확인
        if (item.processingResult?.request_id) serverIds.add(item.processingResult.request_id);
      });

      // 로컬 데이터 중 서버에 이미 존재하는 것(이미 업로드됨)은 제외하고 순수 로컬 데이터만 남김
      const pureLocalResults = localResults.filter((item) => {
        const reqId = item.requestId ?? item.processingResult?.request_id;
        // 로컬 항목에 requestId가 있고 그것이 서버 목록에도 있다면 -> 이미 서버에 있는 것으므로 제외
        if (reqId && serverIds.has(reqId)) {
          return false;
        }
        return true;
      });

      // 5. 최종 리스트 생성 (서버 데이터 + 순수 로컬 데이터)
      const merged = [...pureLocalResults, ...serverRecords];

      // 최신순 정렬 (createdAt 기준)
      merged.sort((a, b) => {
        const aTime = a?.createdAt ? new Date(a.createdAt).getTime() : 0;
        const bTime = b?.createdAt ? new Date(b.createdAt).getTime() : 0;
        return bTime - aTime;
      });

      setResults(merged);

      try {
        const statusList = await fetchStepThreeStatuses();
        const statusMap = (statusList || []).reduce((accumulator, item) => {
          if (item?.id) {
            accumulator[item.id] = item;
          }
          return accumulator;
        }, {});
        setStepThreeStatuses(statusMap);
        return { merged, statusMap };
      } catch (statusError) {
        console.error('Failed to load step two statuses', statusError);
        setStepThreeStatuses({});
        return { merged, statusMap: {} };
      }
    } finally {
      if (showLoader) {
        setIsLoadingResults(false);
      }
    }
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

  const existingFloors = useMemo(() => {
    return sortedResults.map((result) => ({
      floorLabel: recordFloorLabel(result),
      floorValue: recordFloorValue(result),
    }));
  }, [sortedResults]);

  const handleSelectFile = async ({ file, floorLabel, floorValue }) => {
    if (!file || !floorLabel) {
      return;
    }
    setError(null);
    setIsUploading(true);
    try {
      let floorSummaries = [];
      try {
        floorSummaries = await fetchFloorPlanFloors();
      } catch (floorError) {
        console.error('Failed to fetch floor summaries before upload', floorError);
        // eslint-disable-next-line no-alert
        alert('서버에서 층 정보를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.');
        setError('서버에서 층 정보를 불러오지 못했습니다.');
        return;
      }

      if (hasDuplicateFloor(floorSummaries, floorLabel, floorValue)) {
        // eslint-disable-next-line no-alert
        alert('이미 존재하는 층입니다. 다른 층을 선택해 주세요.');
        setError('이미 존재하는 층입니다. 다른 층을 선택해 주세요.');
        return;
      }

      const result = await uploadFloorPlan(file);
      setUploadData({
        ...result,
        floorLabel,
        floorValue,
      });
      navigate('/admin/editor');
    } catch (uploadError) {
      console.error(uploadError);
      setError('업로드에 실패했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleSelectStepThree = (stepOneResult, targetStage = 'base') => {
    if (!stepOneResult) {
      return;
    }
    const requestId =
      stepOneResult?.processingResult?.request_id ?? stepOneResult?.requestId ?? stepOneResult?.id ?? null;
    if (!requestId) {
      // eslint-disable-next-line no-alert
      alert('요청 ID를 찾을 수 없어 3단계로 이동할 수 없습니다.');
      return;
    }
    const hasGraph = Boolean(stepOneResult?.processingResult?.metadata?.graph_summary);
    if (!hasGraph) {
      // eslint-disable-next-line no-alert
      alert('그래프가 아직 생성되지 않았습니다. 먼저 2단계(그래프 편집)에서 그래프를 저장해 주세요.');
      return;
    }
    navigate(`/admin/step-three/${requestId}`, {
      state: {
        targetStage,
        stepOneResult,
        requestId,
      },
    });
  };

  const handleEditStepOne = async (stepOneResult) => {
    if (!stepOneResult) {
      return;
    }
    setError(null);
    try {
      await loadStepOneResultForEdit(stepOneResult);
      navigate('/admin/editor');
    } catch (editError) {
      console.error('Failed to prepare editor payload', editError);
      setError('편집 데이터를 불러오는 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.');
    }
  };

  const handleOpenGraphEditor = async (stepOneResult) => {
    if (!stepOneResult) {
      return;
    }

    let resolvedStepOne = stepOneResult;
    let targetRequestId =
      stepOneResult?.processingResult?.request_id ?? stepOneResult?.requestId ?? stepOneResult?.id ?? null;
    const hasGraphSummary = Boolean(stepOneResult?.processingResult?.metadata?.graph_summary);

    if (!hasGraphSummary) {
      setGraphOpeningStepOneId(stepOneResult.id);
      try {
        const { stepOneResult: updatedStepOne, processingResult } = await processStepOneForStepTwo({
          stepOneResult,
        });
        resolvedStepOne = updatedStepOne;
        targetRequestId =
          processingResult?.request_id ??
          updatedStepOne?.processingResult?.request_id ??
          updatedStepOne?.requestId ??
          null;
        setResults((prev) => prev.map((item) => (item.id === updatedStepOne.id ? updatedStepOne : item)));
      } catch (error) {
        console.error('Failed to process floor plan before opening graph editor', error);
        const isNetworkError = error?.message === 'Network Error' || error?.code === 'ERR_NETWORK';
        const detailMessage =
          error?.response?.data?.detail ??
          (isNetworkError
            ? '화면 경계가 겹쳐 방이 명확히 구분되지 않아 그래프를 만들 수 없습니다. 1단계에서 벽과 방을 다시 정리해 주세요.'
            : (error?.message ?? '그래프를 생성하지 못했습니다. 1단계를 다시 확인해 주세요.'));
        alert(detailMessage);
        return;
      } finally {
        setGraphOpeningStepOneId(null);
      }
    }

    if (!targetRequestId) {
      alert('그래프 편집을 진행할 수 있는 요청 정보를 찾지 못했습니다. 1단계 결과를 다시 확인해 주세요.');
      return;
    }

    navigate(`/admin/graph/${targetRequestId}`, {
      state: {
        stepOneResult: resolvedStepOne,
        stepOneId: resolvedStepOne?.id ?? null,
      },
    });
  };

  const handleDeleteStepOne = async (stepOneResult) => {
    if (!stepOneResult) {
      return;
    }
    const targetRequestId = stepOneResult?.processingResult?.request_id ?? stepOneResult?.requestId ?? null;
    if (!targetRequestId) {
      // eslint-disable-next-line no-alert
      alert('삭제할 요청 정보를 찾지 못했습니다. 다시 확인해 주세요.');
      return;
    }
    setDeletingRequestId(targetRequestId);
    try {
      await deleteStoredFloorPlan({
        requestId: targetRequestId,
        stepOneId: stepOneResult?.id ?? null,
      });
      deleteStepOneResult(stepOneResult?.id);
      setResults((prev) => prev.filter((item) => item.id !== stepOneResult.id));
      setStepThreeStatuses((prev) => {
        if (!prev) {
          return {};
        }
        const next = { ...prev };
        delete next[stepOneResult.id];
        return next;
      });
    } catch (deleteError) {
      console.error('Failed to delete stored floor plan', deleteError);
      const isNetworkError = deleteError?.message === 'Network Error' || deleteError?.code === 'ERR_NETWORK';
      const detailMessage =
        deleteError?.response?.data?.detail ??
        (isNetworkError ? '네트워크 오류로 삭제에 실패했습니다. 잠시 후 다시 시도해 주세요.' : deleteError?.message) ??
        '도면을 삭제하지 못했습니다. 잠시 후 다시 시도해 주세요.';
      // eslint-disable-next-line no-alert
      alert(detailMessage);
    } finally {
      setDeletingRequestId(null);
    }
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
        <FloorPlanUpload
          onSelectFile={handleSelectFile}
          isUploading={isUploading}
          error={error}
          existingFloors={existingFloors}
        />
      </div>
    );
  }

  return (
    <div className={layoutStyles.fillContainer}>
      <div className={styles.wrapper}>
        <section className={styles.uploadSection}>
          <FloorPlanUpload
            onSelectFile={handleSelectFile}
            isUploading={isUploading}
            error={error}
            existingFloors={existingFloors}
          />
        </section>

        <section className={styles.resultsSection}>
          <div className={styles.resultsHeader}>
            <div>
              <h2 className={styles.resultsTitle}>저장된 도면 ({sortedResults.length})</h2>
              <p className={styles.resultsSubtitle}>
                다시 검수하거나 3단계 메타데이터 입력을 진행할 도면을 선택하세요.
              </p>
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
                onSelectStepThree={handleSelectStepThree}
                onOpenGraph={handleOpenGraphEditor}
                onEditStepOne={handleEditStepOne}
                isGraphOpening={graphOpeningStepOneId === result.id}
                stepThreeStatus={stepThreeStatuses[result.id]}
                onDelete={handleDeleteStepOne}
                isDeleting={deletingRequestId === (result?.processingResult?.request_id ?? result?.requestId ?? null)}
              />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
};

export default AdminUploadPage;
