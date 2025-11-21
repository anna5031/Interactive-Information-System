import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Plus, Save, Trash2 } from 'lucide-react';
import PropTypes from 'prop-types';
import StepTwoCanvas from '../components/stepTwo/StepTwoCanvas';
import { getStoredStepOneResultById, saveStepOneResult } from '../api/stepOneResults';
import { fetchStepThreeResultById, saveStepThree } from '../api/stepThreeResults';
import { fetchProcessingResultById, fetchStoredFloorPlanByStepOneId } from '../api/floorPlans';
import { buildStepOneRecordFromProcessingData, buildStepOneRecordFromStoredResult } from '../utils/processingResult';
import styles from './AdminStepThreePage.module.css';

const buildRoomDisplayLabel = (name, number) => {
  const parts = [name, number].map((value) => value?.trim()).filter(Boolean);
  return parts.length > 0 ? parts.join(', ') : '';
};

const sanitizeEntries = (entries) => {
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries
    .map((entry) => ({
      key: entry?.key?.trim() ?? '',
      value: entry?.value?.trim() ?? '',
    }))
    .filter((entry) => entry.key || entry.value);
};

const KeyValueEditor = ({ entries, onChange, addButtonLabel }) => {
  const safeEntries = Array.isArray(entries) ? entries : [];

  const handleChange = (index, field, value) => {
    onChange(safeEntries.map((entry, entryIndex) => (entryIndex === index ? { ...entry, [field]: value } : entry)));
  };

  const handleAdd = () => {
    onChange([...safeEntries, { key: '', value: '' }]);
  };

  const handleRemove = (index) => {
    onChange(safeEntries.filter((_, entryIndex) => entryIndex !== index));
  };

  return (
    <div className={styles.keyValueEditor}>
      {safeEntries.length === 0 && <p className={styles.keyValueHint}>추가 정보를 입력해 주세요.</p>}
      {safeEntries.map((entry, index) => (
        <div key={`entry-${index}`} className={styles.keyValueRow}>
          <input
            type='text'
            placeholder='key (예: 용도)'
            value={entry.key}
            onChange={(event) => handleChange(index, 'key', event.target.value)}
          />
          <input
            type='text'
            placeholder='value (예: 세미나실)'
            value={entry.value}
            onChange={(event) => handleChange(index, 'value', event.target.value)}
          />
          <button type='button' className={styles.iconButton} onClick={() => handleRemove(index)}>
            <Trash2 size={16} />
          </button>
        </div>
      ))}
      <button type='button' className={styles.addButton} onClick={handleAdd}>
        <Plus size={16} />
        {addButtonLabel}
      </button>
    </div>
  );
};

KeyValueEditor.propTypes = {
  entries: PropTypes.arrayOf(
    PropTypes.shape({
      key: PropTypes.string,
      value: PropTypes.string,
    })
  ),
  onChange: PropTypes.func.isRequired,
  addButtonLabel: PropTypes.string,
};

KeyValueEditor.defaultProps = {
  entries: [],
  addButtonLabel: '필드 추가',
};

const AdminStepThreePage = () => {
  const { stepOneId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const locationTargetStage = location.state?.targetStage ?? null;
  const locationStepOneResult = location.state?.stepOneResult ?? null;
  const locationRequestId = location.state?.requestId ?? null;

  const [stepOneResult, setStepOneResult] = useState(null);
  const [stepOneBoxes, setStepOneBoxes] = useState([]);
  const [stepOnePoints, setStepOnePoints] = useState([]);
  const [stepOneLines, setStepOneLines] = useState([]);
  const [roomsState, setRoomsState] = useState([]);
  const [doorsState, setDoorsState] = useState([]);
  const [stage, setStage] = useState('base');
  const [isBaseSaving, setIsBaseSaving] = useState(false);
  const [isDetailSaving, setIsDetailSaving] = useState(false);
  const [detailSaveError, setDetailSaveError] = useState(null);
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [processingData, setProcessingData] = useState(null);
  const [isProcessingLoading, setIsProcessingLoading] = useState(false);
  const [processingError, setProcessingError] = useState(null);
  const [showRoomLabels, setShowRoomLabels] = useState(true);
  const [showDoorLabels, setShowDoorLabels] = useState(true);
  const [stepThreeRecord, setStepThreeRecord] = useState(null);
  const [isStepThreeLoading, setIsStepThreeLoading] = useState(false);
  const [stepThreeError, setStepThreeError] = useState(null);
  const [isStepThreeFetchComplete, setIsStepThreeFetchComplete] = useState(false);

  const targetStageRef = useRef(null);
  const hasConsumedTargetStage = useRef(false);
  const pageRef = useRef(null);
  const [pageHeight, setPageHeight] = useState(null);
  const roomListRefs = useRef(new Map());
  const doorListRefs = useRef(new Map());
  const roomDetailsRefs = useRef(new Map());
  const doorDetailsRefs = useRef(new Map());
  const hasHydratedStepThree = useRef(false);
  const lastSyncedPayloadRef = useRef(null);
  const hydratedProcessingRequestsRef = useRef(new Set());

  const latestProcessingRequestId = useMemo(() => {
    return stepOneResult?.processingResult?.request_id ?? locationRequestId ?? stepOneResult?.requestId ?? null;
  }, [stepOneResult?.processingResult?.request_id, locationRequestId, stepOneResult?.requestId]);

  const processingRequestId = latestProcessingRequestId ?? stepThreeRecord?.requestId ?? null;
  const resolvedStepThreeId = stepOneResult?.id ?? stepOneId ?? null;

  const updatePageHeight = useCallback(() => {
    const element = pageRef.current;
    if (!element) {
      return;
    }
    const rect = element.getBoundingClientRect();
    const available = window.innerHeight - rect.top;
    if (!Number.isFinite(available)) {
      return;
    }
    const safeHeight = Math.max(available, 0);
    setPageHeight((prev) => {
      if (prev == null || Math.abs(prev - safeHeight) > 1) {
        return safeHeight;
      }
      return prev;
    });
  }, []);

  const registerRoomListRef = useCallback((nodeId, element) => {
    if (!roomListRefs.current) {
      roomListRefs.current = new Map();
    }
    if (element) {
      roomListRefs.current.set(nodeId, element);
    } else {
      roomListRefs.current.delete(nodeId);
    }
  }, []);

  const registerDoorListRef = useCallback((nodeId, element) => {
    if (!doorListRefs.current) {
      doorListRefs.current = new Map();
    }
    if (element) {
      doorListRefs.current.set(nodeId, element);
    } else {
      doorListRefs.current.delete(nodeId);
    }
  }, []);

  const registerRoomDetailsRef = useCallback((nodeId, element) => {
    if (!roomDetailsRefs.current) {
      roomDetailsRefs.current = new Map();
    }
    if (element) {
      roomDetailsRefs.current.set(nodeId, element);
    } else {
      roomDetailsRefs.current.delete(nodeId);
    }
  }, []);

  const registerDoorDetailsRef = useCallback((nodeId, element) => {
    if (!doorDetailsRefs.current) {
      doorDetailsRefs.current = new Map();
    }
    if (element) {
      doorDetailsRefs.current.set(nodeId, element);
    } else {
      doorDetailsRefs.current.delete(nodeId);
    }
  }, []);

  useLayoutEffect(() => {
    updatePageHeight();
    window.addEventListener('resize', updatePageHeight);
    return () => {
      window.removeEventListener('resize', updatePageHeight);
    };
  }, [updatePageHeight]);

  useEffect(() => {
    updatePageHeight();
  }, [stage, roomsState, doorsState, updatePageHeight]);

  useEffect(() => {
    if (locationTargetStage == null || hasConsumedTargetStage.current) {
      return;
    }
    targetStageRef.current = locationTargetStage;
    hasConsumedTargetStage.current = true;
  }, [locationTargetStage]);

  const pageStyle = useMemo(
    () => (pageHeight ? { height: pageHeight, maxHeight: pageHeight, minHeight: pageHeight } : undefined),
    [pageHeight]
  );

  useEffect(() => {
    let cancelled = false;

    const hydrateStepOneResult = async () => {
      setRoomsState([]);
      setDoorsState([]);
      setStage('base');
      setSelectedEntity(null);
      setProcessingData(null);
      setProcessingError(null);
      setIsProcessingLoading(true);
      setStepThreeRecord(null);
      setIsStepThreeLoading(true);
      setStepThreeError(null);
      setIsStepThreeFetchComplete(false);
      hasHydratedStepThree.current = false;
      hasConsumedTargetStage.current = false;
      if (typeof locationTargetStage === 'string') {
        targetStageRef.current = locationTargetStage;
      } else {
        targetStageRef.current = null;
      }

      let resolved = null;

      if (stepOneId) {
        resolved = getStoredStepOneResultById(stepOneId);
      }

      if (!resolved && locationStepOneResult) {
        resolved = locationStepOneResult;
      }

      if (!resolved && stepOneId) {
        try {
          const summary = await fetchStoredFloorPlanByStepOneId(stepOneId);
          if (summary) {
            resolved = buildStepOneRecordFromStoredResult(summary);
            if (resolved) {
              await saveStepOneResult({
                sourceOriginalId: resolved.id,
                createdAt: resolved.createdAt,
                fileName: resolved.fileName,
                filePath: resolved.filePath,
                imageUrl: resolved.metadata?.imageUrl ?? resolved.imageUrl ?? null,
                imageDataUrl: resolved.imageDataUrl ?? null,
                metadata: resolved.metadata,
                objectDetection: resolved.objectDetection,
                wall: resolved.wall,
                door: resolved.door,
                processingResult: resolved.processingResult,
                origin: 'server',
                requestId: resolved.requestId ?? null,
              });
            }
          }
        } catch (error) {
          console.error('Failed to fetch stored floor plan by Step One ID', error);
        }
      }

      if (!resolved && locationRequestId) {
        try {
          const processingPayload = await fetchProcessingResultById(locationRequestId);
          if (processingPayload) {
            resolved = buildStepOneRecordFromProcessingData(processingPayload, {
              stepOneId,
            });
            if (resolved) {
              await saveStepOneResult({
                sourceOriginalId: resolved.id,
                createdAt: resolved.createdAt,
                fileName: resolved.fileName,
                filePath: resolved.filePath,
                imageUrl: resolved.metadata?.imageUrl ?? resolved.imageUrl ?? null,
                imageDataUrl: resolved.imageDataUrl ?? null,
                metadata: resolved.metadata,
                objectDetection: resolved.objectDetection,
                wall: resolved.wall,
                door: resolved.door,
                processingResult: resolved.processingResult,
                origin: 'server',
                requestId: resolved.requestId ?? null,
              });
            }
          }
        } catch (error) {
          console.error('Failed to hydrate Step One record from processing data', error);
        }
      }

      if (!resolved) {
        if (!cancelled) {
          navigate('/admin/upload', { replace: true });
        }
        return;
      }

      if (cancelled) {
        return;
      }

      setStepOneResult(resolved);

      const boxes = Array.isArray(resolved?.objectDetection?.boxes) ? resolved.objectDetection.boxes : [];
      const points = Array.isArray(resolved?.door?.points) ? resolved.door.points : [];
      const lines = Array.isArray(resolved?.wall?.lines) ? resolved.wall.lines : [];

      setStepOneBoxes(boxes);
      setStepOnePoints(points);
      setStepOneLines(lines);

      setIsProcessingLoading(true);
    };

    hydrateStepOneResult();

    return () => {
      cancelled = true;
    };
  }, [stepOneId, navigate, locationTargetStage, locationStepOneResult, locationRequestId]);

  useEffect(() => {
    if (!stepOneResult) {
      setIsProcessingLoading(false);
      return;
    }

    if (!isStepThreeFetchComplete) {
      return;
    }

    const requestId = processingRequestId ?? null;
    if (!requestId) {
      // eslint-disable-next-line no-alert
      alert('그래프 요청 정보를 찾을 수 없습니다. 2단계에서 그래프를 먼저 생성해 주세요.');
      navigate('/admin/upload', { replace: true });
      return;
    }

    const hasCompleteProcessing = Boolean(stepOneResult?.processingResult?.metadata?.graph_summary);
    if (!hasCompleteProcessing) {
      // eslint-disable-next-line no-alert
      alert('그래프가 아직 생성되지 않았습니다. 먼저 2단계(그래프 편집)에서 그래프를 저장해 주세요.');
      navigate('/admin/upload', { replace: true });
      return;
    }

    let cancelled = false;

    const applyProcessingResult = (processingResult) => {
      if (cancelled) {
        return;
      }
      setProcessingData(processingResult ?? null);
      if (processingResult?.request_id) {
        hydratedProcessingRequestsRef.current.add(processingResult.request_id);
      }
    };

    if (hydratedProcessingRequestsRef.current.has(requestId) && stepOneResult.processingResult) {
      applyProcessingResult(stepOneResult.processingResult);
      setIsProcessingLoading(false);
      return;
    }

    const loadProcessing = async () => {
      setIsProcessingLoading(true);
      setProcessingError(null);

      try {
        const data = await fetchProcessingResultById(requestId);
        if (cancelled) {
          return;
        }

        const resolvedData = data ?? stepOneResult.processingResult ?? null;
        applyProcessingResult(resolvedData);

        const needsHydration =
          !stepOneResult.processingResult ||
          !stepOneResult.processingResult?.objects ||
          !stepOneResult.processingResult?.graph;

        if (data && stepOneResult?.id && needsHydration) {
          const rebuilt = buildStepOneRecordFromProcessingData(data, {
            stepOneId: stepOneResult.id,
            sourceImagePath: stepOneResult?.metadata?.fileName ?? stepOneResult?.metadata?.sourceImagePath ?? null,
          });
          if (rebuilt) {
            const mergedRecord = {
              ...rebuilt,
              origin: stepOneResult?.origin ?? 'server',
            };
            setStepOneResult(mergedRecord);
            try {
              await saveStepOneResult(mergedRecord);
            } catch (persistError) {
              console.warn('Failed to persist hydrated processing result', persistError);
            }
          }
        }
      } catch (loadError) {
        console.error('Failed to load processing result', loadError);
        if (!cancelled) {
          setProcessingError('그래프 데이터를 불러오지 못했습니다. 다시 시도해 주세요.');
        }
      } finally {
        if (!cancelled) {
          setIsProcessingLoading(false);
        }
      }
    };

    loadProcessing();
    return () => {
      cancelled = true;
    };
  }, [stepOneResult, processingRequestId, navigate, isStepThreeFetchComplete]);

  useEffect(() => {
    if (!resolvedStepThreeId) {
      setIsStepThreeLoading(false);
      setIsStepThreeFetchComplete(true);
      return;
    }

    let cancelled = false;
    setIsStepThreeLoading(true);
    setStepThreeError(null);
    setIsStepThreeFetchComplete(false);

    fetchStepThreeResultById(resolvedStepThreeId)
      .then((record) => {
        if (cancelled) {
          return;
        }
        setStepThreeRecord(record ?? null);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        if (error?.response?.status === 404) {
          setStepThreeRecord(null);
          return;
        }
        console.error('Failed to fetch step three result', error);
        setStepThreeError('3단계 데이터를 불러오지 못했습니다.');
        setStepThreeRecord(null);
      })
      .finally(() => {
        if (!cancelled) {
          setIsStepThreeLoading(false);
          setIsStepThreeFetchComplete(true);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [resolvedStepThreeId, stepOneResult?.id]);

  const processingObjects = useMemo(() => {
    const objects = processingData?.objects ?? stepOneResult?.processingResult?.objects ?? null;
    if (!objects) {
      return null;
    }
    return {
      rooms: Array.isArray(objects.rooms) ? objects.rooms : [],
      doors: Array.isArray(objects.doors) ? objects.doors : [],
      stairs: Array.isArray(objects.stairs) ? objects.stairs : [],
      elevators: Array.isArray(objects.elevators) ? objects.elevators : [],
    };
  }, [processingData, stepOneResult]);

  const processingGraph = useMemo(() => {
    const graph = processingData?.graph ?? stepOneResult?.processingResult?.graph ?? null;
    if (!graph) {
      return { nodes: [], edges: [] };
    }
    return {
      nodes: Array.isArray(graph.nodes) ? graph.nodes : [],
      edges: Array.isArray(graph.edges) ? graph.edges : [],
    };
  }, [processingData, stepOneResult]);

  const stepOneFloorLabel = useMemo(() => {
    return (
      stepOneResult?.metadata?.floorLabel ??
      stepOneResult?.floorLabel ??
      stepThreeRecord?.floorLabel ??
      null
    );
  }, [stepOneResult, stepThreeRecord]);

  const stepOneFloorValue = useMemo(() => {
    return (
      stepOneResult?.metadata?.floorValue ??
      stepOneResult?.floorValue ??
      stepThreeRecord?.floorValue ??
      null
    );
  }, [stepOneResult, stepThreeRecord]);

  const buildStepThreePayload = useCallback(() => {
    const floorLabel = stepOneFloorLabel;
    const floorValue = stepOneFloorValue;
    const storedRoomsMap = new Map((stepThreeRecord?.rooms ?? []).map((room) => [room.nodeId, room]));
    const storedDoorsMap = new Map((stepThreeRecord?.doors ?? []).map((door) => [door.nodeId, door]));
    return {
      stepOneId: stepOneResult?.id,
      requestId: processingRequestId,
      floorLabel,
      floorValue,
      rooms: roomsState.map((room) => {
        const stored = storedRoomsMap.get(room.nodeId);
        const fallbackName = (stored?.name ?? '').trim();
        const fallbackNumber = (stored?.number ?? '').trim();
        const sanitizedExtra = sanitizeEntries(room.extra);
        const shouldReuseStoredExtra =
          !hasHydratedStepThree.current &&
          sanitizedExtra.length === 0 &&
          Array.isArray(stored?.extra) &&
          stored.extra.length > 0;
        return {
          nodeId: room.nodeId,
          graphNodeId: stored?.graphNodeId ?? room.nodeId,
          name: (room.name ?? '').trim() || fallbackName,
          number: (room.number ?? '').trim() || fallbackNumber,
          extra: shouldReuseStoredExtra ? sanitizeEntries(stored.extra) : sanitizedExtra,
        };
      }),
      doors: doorsState.map((door) => {
        const stored = storedDoorsMap.get(door.nodeId);
        const fallbackType = (stored?.type ?? '').trim();
        const fallbackCustom = (stored?.customType ?? '').trim();
        const sanitizedExtra = sanitizeEntries(door.extra);
        const shouldReuseStoredExtra =
          !hasHydratedStepThree.current &&
          sanitizedExtra.length === 0 &&
          Array.isArray(stored?.extra) &&
          stored.extra.length > 0;
        return {
          nodeId: door.nodeId,
          graphNodeId: stored?.graphNodeId ?? door.nodeId,
          type: (door.type ?? '').trim() || fallbackType,
          customType: (door.customType ?? '').trim() || fallbackCustom,
          extra: shouldReuseStoredExtra ? sanitizeEntries(stored.extra) : sanitizedExtra,
        };
      }),
    };
  }, [
    roomsState,
    doorsState,
    stepOneResult?.id,
    stepOneFloorLabel,
    stepOneFloorValue,
    processingRequestId,
    stepThreeRecord,
  ]);

  const serializePayloadForSync = useCallback((payload) => {
    if (!payload) {
      return null;
    }
    return JSON.stringify({
      requestId: payload.requestId ?? null,
      floorLabel: payload.floorLabel ?? null,
      floorValue: payload.floorValue ?? null,
      rooms: payload.rooms,
      doors: payload.doors,
    });
  }, []);

  const persistStepThreeDraft = useCallback(
    async ({ force = false } = {}) => {
      if (!stepOneResult?.id) {
        return null;
      }

      const hasExistingStoredData = Boolean(
        (Array.isArray(stepThreeRecord?.rooms) && stepThreeRecord.rooms.length > 0) ||
          (Array.isArray(stepThreeRecord?.doors) && stepThreeRecord.doors.length > 0)
      );

      if (!roomsState.length && !doorsState.length && hasExistingStoredData) {
        // 그래프 엔티티를 아직 불러오기 전에 저장이 호출되면 기존 데이터를 덮어쓰지 않도록 방어
        return stepThreeRecord;
      }

      const payload = buildStepThreePayload();
      const signature = serializePayloadForSync(payload);
      if (!force && signature === lastSyncedPayloadRef.current) {
        return stepThreeRecord;
      }

      try {
        const record = await saveStepThree(payload);
        setStepThreeRecord(record ?? null);
        lastSyncedPayloadRef.current = signature;
        return record;
      } catch (error) {
        console.error('Failed to save step three data', error);
        throw error;
      }
    },
    [
      buildStepThreePayload,
      serializePayloadForSync,
      stepOneResult?.id,
      stepThreeRecord,
      roomsState.length,
      doorsState.length,
    ]
  );

  const roomsById = useMemo(() => {
    const map = new Map();
    (processingObjects?.rooms ?? []).forEach((room) => {
      if (typeof room?.id === 'number') {
        map.set(room.id, room);
      }
    });
    return map;
  }, [processingObjects]);

  const doorsById = useMemo(() => {
    const map = new Map();
    (processingObjects?.doors ?? []).forEach((door) => {
      if (typeof door?.id === 'number') {
        map.set(door.id, door);
      }
    });
    return map;
  }, [processingObjects]);

  const polygonToString = (points) =>
    Array.isArray(points) && points.length > 1 ? points.map((pt) => pt.join(',')).join(' ') : null;

  const imageSize = useMemo(() => {
    if (processingData?.image_size?.width && processingData.image_size?.height) {
      return processingData.image_size;
    }
    if (stepOneResult?.metadata?.imageWidth && stepOneResult?.metadata?.imageHeight) {
      return {
        width: stepOneResult.metadata.imageWidth,
        height: stepOneResult.metadata.imageHeight,
      };
    }
    return { width: 1000, height: 1000 };
  }, [processingData, stepOneResult]);

  const convertBoxToPolygon = useCallback(
    (box) => {
      if (!box || !Number.isFinite(box.x) || !Number.isFinite(box.y)) {
        return null;
      }
      const x = Number(box.x) * imageSize.width;
      const y = Number(box.y) * imageSize.height;
      const widthPx = Number(box.width ?? 0) * imageSize.width;
      const heightPx = Number(box.height ?? 0) * imageSize.height;
      if (!Number.isFinite(widthPx) || !Number.isFinite(heightPx)) {
        return null;
      }
      return [
        [x, y],
        [x + widthPx, y],
        [x + widthPx, y + heightPx],
        [x, y + heightPx],
      ];
    },
    [imageSize]
  );

  const roomsFromBoxes = useMemo(
    () =>
      stepOneBoxes
        .filter((box) => String(box.labelId) === '2')
        .map((box, index) => {
          const polygonPoints = convertBoxToPolygon(box);
          if (!polygonPoints) {
            return null;
          }
          const centroid = [
            polygonPoints[0][0] + (polygonPoints[1][0] - polygonPoints[0][0]) / 2,
            polygonPoints[0][1] + (polygonPoints[2][1] - polygonPoints[0][1]) / 2,
          ];
          return {
            nodeId: box.id,
            index: index + 1,
            label: box.id,
            polygon: polygonToString(polygonPoints),
            labelPosition: centroid,
            geometry: {
              polygon: polygonPoints,
              centroid,
              sourceId: box.id,
              object: null,
            },
            object: null,
          };
        })
        .filter(Boolean),
    [stepOneBoxes, convertBoxToPolygon]
  );

  const doorsFromPoints = useMemo(
    () =>
      stepOnePoints
        .filter((point) => String(point.labelId) === '0')
        .map((point, index) => {
          const centroid = [point.x * imageSize.width, point.y * imageSize.height];
          return {
            nodeId: point.id,
            index: index + 1,
            label: point.id,
            polygon: null,
            centroid,
            geometry: {
              polygon: null,
              centroid,
              point,
              sourceId: point.id,
              object: null,
            },
            object: null,
          };
        }),
    [stepOnePoints, imageSize]
  );

  const roomsFromBoxesById = useMemo(() => {
    const map = new Map();
    roomsFromBoxes.forEach((room) => {
      const sourceId = room?.geometry?.sourceId ?? room.nodeId;
      map.set(sourceId, room);
    });
    return map;
  }, [roomsFromBoxes]);

  const doorsFromPointsById = useMemo(() => {
    const map = new Map();
    doorsFromPoints.forEach((door) => {
      const sourceId = door?.geometry?.sourceId ?? door.nodeId;
      map.set(sourceId, door);
    });
    return map;
  }, [doorsFromPoints]);

  const roomEntities = useMemo(() => {
    const graphRooms = (processingGraph?.nodes ?? [])
      .filter((node) => node.type === 'room')
      .slice()
      .sort((a, b) => {
        const aId = Number.parseInt(String(a.id).replace('room_', ''), 10);
        const bId = Number.parseInt(String(b.id).replace('room_', ''), 10);
        if (!Number.isNaN(aId) && !Number.isNaN(bId)) {
          return aId - bId;
        }
        return String(a.id).localeCompare(String(b.id));
      });

    if (graphRooms.length > 0) {
      return graphRooms
        .map((node, index) => {
          const attrs = node?.attributes ?? {};
          const numericId =
            typeof attrs?.rooms_id === 'number'
              ? attrs.rooms_id
              : typeof attrs?.room_id === 'number'
                ? attrs.room_id
                : null;

          let sourceId = attrs?.source_id ?? null;
          if (!sourceId && numericId != null) {
            const graphRoomObject = roomsById.get(numericId);
            if (graphRoomObject?.source_id) {
              sourceId = graphRoomObject.source_id;
            }
          }

          let base = sourceId ? roomsFromBoxesById.get(sourceId) : null;
          if (!base) {
            base = roomsFromBoxes[index] ?? null;
          }

          if (!base) {
            return null;
          }

          const geometry = base.geometry ?? null;
          const label = node.id;

          return {
            nodeId: node.id,
            index: index + 1,
            label,
            polygon: base.polygon,
            labelPosition: base.labelPosition,
            geometry,
            object: roomsById.get(numericId ?? -1) ?? base.object ?? null,
          };
        })
        .filter(Boolean);
    }

    return roomsFromBoxes.map((room, index) => ({
      nodeId: room.nodeId,
      index: index + 1,
      label: room.geometry?.sourceId ?? room.nodeId,
      polygon: room.polygon,
      labelPosition: room.labelPosition,
      geometry: room.geometry,
      object: room.object,
    }));
  }, [processingGraph, roomsById, roomsFromBoxes, roomsFromBoxesById]);

  const doorEntities = useMemo(() => {
    const graphDoors = (processingGraph?.nodes ?? [])
      .filter((node) => node.type === 'door')
      .slice()
      .sort((a, b) => {
        const aId = Number.parseInt(String(a.id).replace('door_', ''), 10);
        const bId = Number.parseInt(String(b.id).replace('door_', ''), 10);
        if (!Number.isNaN(aId) && !Number.isNaN(bId)) {
          return aId - bId;
        }
        return String(a.id).localeCompare(String(b.id));
      });

    if (graphDoors.length > 0) {
      return graphDoors.map((node, index) => {
        const attrs = node?.attributes ?? {};
        const numericId =
          typeof attrs?.door_id === 'number'
            ? attrs.door_id
            : typeof attrs?.doors_id === 'number'
              ? attrs.doors_id
              : null;

        let sourceId = attrs?.source_id ?? null;
        if (!sourceId && numericId != null) {
          const doorObj = doorsById.get(numericId);
          if (doorObj?.source_id) {
            sourceId = doorObj.source_id;
          }
        }

        let base = sourceId ? doorsFromPointsById.get(sourceId) : null;
        if (!base) {
          base = doorsFromPoints[index] ?? null;
        }

        if (!base) {
          return null;
        }

        const geometry = base.geometry ?? null;
        const label = geometry?.sourceId ?? sourceId ?? node.id;

        return {
          nodeId: node.id,
          index: index + 1,
          label,
          polygon: base.polygon,
          centroid: base.centroid,
          geometry,
          object: doorsById.get(numericId ?? -1) ?? base.object ?? null,
        };
      });
    }

    return doorsFromPoints;
  }, [processingGraph, doorsById, doorsFromPoints, doorsFromPointsById]);

  useEffect(() => {
    if (!stepThreeRecord) {
      lastSyncedPayloadRef.current = null;
      hasHydratedStepThree.current = false;
      return;
    }

    if (hasHydratedStepThree.current) {
      return;
    }

    if (roomEntities.length === 0 && doorEntities.length === 0) {
      return;
    }

    const storedRooms = Array.isArray(stepThreeRecord?.rooms) ? stepThreeRecord.rooms : [];
    const storedDoors = Array.isArray(stepThreeRecord?.doors) ? stepThreeRecord.doors : [];
    const hasStoredData = storedRooms.length > 0 || storedDoors.length > 0;

    if (hasStoredData) {
      const currentRoomIds = new Set(roomEntities.map((r) => r.nodeId));
      const currentDoorIds = new Set(doorEntities.map((d) => d.nodeId));

      // 저장된 룸 ID 중 하나라도 현재 화면에 존재하는지 확인
      const isRoomMatch = storedRooms.length === 0 || storedRooms.some((r) => currentRoomIds.has(r.nodeId));
      // 저장된 도어 ID 중 하나라도 현재 화면에 존재하는지 확인
      const isDoorMatch = storedDoors.length === 0 || storedDoors.some((d) => currentDoorIds.has(d.nodeId));

      // 매칭되는 ID가 전혀 없다면 그래프 로딩 대기 (return)
      if (!isRoomMatch && !isDoorMatch) {
        return;
      }
    }
    const storedRoomsMap = new Map(storedRooms.map((item) => [item.nodeId, item]));
    const roomEntityNodes = roomEntities.map((entity) => entity.nodeId);

    setRoomsState((prev) => {
      const prevMap = new Map(prev.map((item) => [item.nodeId, item]));
      return roomEntityNodes.map((nodeId) => {
        const stored = storedRoomsMap.get(nodeId);
        const existing = prevMap.get(nodeId);
        return {
          nodeId,
          name: stored?.name ?? existing?.name ?? '',
          number: stored?.number ?? existing?.number ?? '',
          extra: Array.isArray(stored?.extra)
            ? stored.extra.map((entry) => ({ key: entry.key ?? '', value: entry.value ?? '' }))
            : (existing?.extra ?? []),
        };
      });
    });

    const storedDoorsMap = new Map(storedDoors.map((item) => [item.nodeId, item]));
    const doorEntityNodes = doorEntities.map((entity) => entity.nodeId);

    setDoorsState((prev) => {
      const prevMap = new Map(prev.map((item) => [item.nodeId, item]));
      return doorEntityNodes.map((nodeId) => {
        const stored = storedDoorsMap.get(nodeId);
        const existing = prevMap.get(nodeId);
        return {
          nodeId,
          type: stored?.type ?? existing?.type ?? '',
          customType: stored?.customType ?? existing?.customType ?? '',
          extra: Array.isArray(stored?.extra)
            ? stored.extra.map((entry) => ({ key: entry.key ?? '', value: entry.value ?? '' }))
            : (existing?.extra ?? []),
        };
      });
    });

    hasHydratedStepThree.current = true;
    const signature = JSON.stringify({
      requestId: stepThreeRecord.requestId ?? null,
      rooms: storedRooms,
      doors: storedDoors,
    });
    lastSyncedPayloadRef.current = signature;
    if (targetStageRef.current === 'details') {
      setStage('details');
      targetStageRef.current = null;
    }
  }, [stepThreeRecord, roomEntities, doorEntities]);

  const wallLineEntities = useMemo(() => {
    if (!Array.isArray(stepOneLines) || stepOneLines.length === 0) {
      return [];
    }
    return stepOneLines
      .map((line, index) => {
        const x1 = Number(line.x1 ?? line[0] ?? 0) * imageSize.width;
        const y1 = Number(line.y1 ?? line[1] ?? 0) * imageSize.height;
        const x2 = Number(line.x2 ?? line[2] ?? 0) * imageSize.width;
        const y2 = Number(line.y2 ?? line[3] ?? 0) * imageSize.height;
        const coords = [x1, y1, x2, y2];
        if (!coords.every((value) => Number.isFinite(value))) {
          return null;
        }
        return {
          id: line.id ?? `wall_line_${index}`,
          x1,
          y1,
          x2,
          y2,
        };
      })
      .filter(Boolean);
  }, [stepOneLines, imageSize]);

  const stairEntitiesFromGraph = useMemo(
    () =>
      (processingObjects?.stairs ?? [])
        .map((stair) => {
          const polygonPoints =
            (Array.isArray(stair?.polygon?.coordinates?.[0]) && stair.polygon.coordinates[0]) ||
            (Array.isArray(stair?.corners) && stair.corners) ||
            null;
          if (!polygonPoints || polygonPoints.length < 3) {
            return null;
          }
          return {
            id: stair?.source_id ?? `stair_${stair?.id}`,
            polygon: polygonToString(polygonPoints),
          };
        })
        .filter(Boolean),
    [processingObjects]
  );

  const stairEntities = useMemo(() => {
    const fromBoxes = stepOneBoxes
      .filter((box) => String(box.labelId) === '3')
      .map((box) => {
        const polygonPoints = convertBoxToPolygon(box);
        return polygonPoints
          ? {
              id: box.id,
              polygon: polygonToString(polygonPoints),
            }
          : null;
      })
      .filter(Boolean);
    return fromBoxes.length > 0 ? fromBoxes : stairEntitiesFromGraph;
  }, [stepOneBoxes, convertBoxToPolygon, stairEntitiesFromGraph]);

  const elevatorEntitiesFromGraph = useMemo(
    () =>
      (processingObjects?.elevators ?? [])
        .map((el) => {
          const polygonPoints =
            (Array.isArray(el?.polygon?.coordinates?.[0]) && el.polygon.coordinates[0]) ||
            (Array.isArray(el?.corners) && el.corners) ||
            null;
          if (!polygonPoints || polygonPoints.length < 3) {
            return null;
          }
          return {
            id: el?.source_id ?? `elevator_${el?.id}`,
            polygon: polygonToString(polygonPoints),
          };
        })
        .filter(Boolean),
    [processingObjects]
  );

  const elevatorEntities = useMemo(() => {
    const fromBoxes = stepOneBoxes
      .filter((box) => String(box.labelId) === '1')
      .map((box) => {
        const polygonPoints = convertBoxToPolygon(box);
        return polygonPoints
          ? {
              id: box.id,
              polygon: polygonToString(polygonPoints),
            }
          : null;
      })
      .filter(Boolean);
    return fromBoxes.length > 0 ? fromBoxes : elevatorEntitiesFromGraph;
  }, [stepOneBoxes, convertBoxToPolygon, elevatorEntitiesFromGraph]);

  const roomEntityMap = useMemo(() => new Map(roomEntities.map((entity) => [entity.nodeId, entity])), [roomEntities]);
  const doorEntityMap = useMemo(() => new Map(doorEntities.map((entity) => [entity.nodeId, entity])), [doorEntities]);

  const scrollEntityIntoView = useCallback(
    (entity) => {
      if (!entity) {
        return;
      }

      const targetMap =
        entity.type === 'room'
          ? stage === 'details'
            ? roomDetailsRefs.current
            : roomListRefs.current
          : stage === 'details'
            ? doorDetailsRefs.current
            : doorListRefs.current;

      const target = targetMap?.get(entity.nodeId) ?? null;
      if (!target || typeof target.scrollIntoView !== 'function') {
        return;
      }

      window.requestAnimationFrame(() => {
        target.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
      });
    },
    [stage]
  );

  const ensureDetailsOpen = useCallback(
    (entity) => {
      if (!entity || stage !== 'details') {
        return;
      }

      const selector =
        entity.type === 'room'
          ? `details[data-room-id="${entity.nodeId}"]`
          : `details[data-door-id="${entity.nodeId}"]`;

      window.requestAnimationFrame(() => {
        const detailsEl = document.querySelector(selector);
        if (detailsEl && !detailsEl.open) {
          detailsEl.open = true;
        }
      });
    },
    [stage]
  );

  useEffect(() => {
    if (roomEntities.length === 0) {
      return;
    }
    setRoomsState((prev) => {
      const ids = roomEntities.map((entity) => entity.nodeId);
      const filtered = prev.filter((room) => ids.includes(room.nodeId));
      const ordered = ids.map((id) => filtered.find((room) => room.nodeId === id)).filter(Boolean);
      const existingIds = new Set(ordered.map((room) => room.nodeId));
      const additions = ids
        .filter((id) => !existingIds.has(id))
        .map((id) => ({
          nodeId: id,
          name: '',
          number: '',
          extra: [],
        }));
      if (additions.length === 0 && ordered.length === prev.length) {
        return prev;
      }
      return [...ordered, ...additions];
    });
  }, [roomEntities]);

  useEffect(() => {
    if (doorEntities.length === 0) {
      return;
    }
    setDoorsState((prev) => {
      const ids = doorEntities.map((entity) => entity.nodeId);
      const filtered = prev.filter((door) => ids.includes(door.nodeId));
      const ordered = ids.map((id) => filtered.find((door) => door.nodeId === id)).filter(Boolean);
      const existingIds = new Set(ordered.map((door) => door.nodeId));
      const additions = ids
        .filter((id) => !existingIds.has(id))
        .map((id) => ({
          nodeId: id,
          type: '',
          customType: '',
          extra: [],
        }));
      if (additions.length === 0 && ordered.length === prev.length) {
        return prev;
      }
      return [...ordered, ...additions];
    });
  }, [doorEntities]);

  useEffect(() => {
    if (!selectedEntity) {
      if (roomEntities.length > 0) {
        setSelectedEntity({ type: 'room', nodeId: roomEntities[0].nodeId });
      } else if (doorEntities.length > 0) {
        setSelectedEntity({ type: 'door', nodeId: doorEntities[0].nodeId });
      }
      return;
    }

    if (selectedEntity.type === 'room' && !roomEntityMap.has(selectedEntity.nodeId)) {
      setSelectedEntity(null);
    } else if (selectedEntity.type === 'door' && !doorEntityMap.has(selectedEntity.nodeId)) {
      setSelectedEntity(null);
    }
  }, [selectedEntity, roomEntities, doorEntities, roomEntityMap, doorEntityMap]);

  const handleSelectEntity = (entity) => {
    setSelectedEntity(entity);
    if (!entity) {
      return;
    }

    scrollEntityIntoView(entity);
    ensureDetailsOpen(entity);
  };

  useEffect(() => {
    if (!selectedEntity) {
      return;
    }
    scrollEntityIntoView(selectedEntity);
    ensureDetailsOpen(selectedEntity);
  }, [selectedEntity, scrollEntityIntoView, ensureDetailsOpen]);

  const handleRoomFieldChange = (nodeId, field, value) => {
    setRoomsState((prev) => prev.map((room) => (room.nodeId === nodeId ? { ...room, [field]: value } : room)));
    setSelectedEntity({ type: 'room', nodeId });
  };

  const handleDoorFieldChange = (nodeId, field, value) => {
    setDoorsState((prev) => prev.map((door) => (door.nodeId === nodeId ? { ...door, [field]: value } : door)));
    setSelectedEntity({ type: 'door', nodeId });
  };

  const handleRoomExtraChange = (nodeId, extra) => {
    setRoomsState((prev) => prev.map((room) => (room.nodeId === nodeId ? { ...room, extra } : room)));
    setSelectedEntity({ type: 'room', nodeId });
  };

  const handleDoorExtraChange = (nodeId, extra) => {
    setDoorsState((prev) => prev.map((door) => (door.nodeId === nodeId ? { ...door, extra } : door)));
    setSelectedEntity({ type: 'door', nodeId });
  };

  const handleBaseSave = async () => {
    const invalidRoom = roomsState.find((room) => !room.name?.trim() && !room.number?.trim());
    if (invalidRoom) {
      setSelectedEntity({ type: 'room', nodeId: invalidRoom.nodeId });
      // eslint-disable-next-line no-alert
      alert('각 방은 “방 이름” 또는 “방 호수” 중 하나 이상을 입력해야 합니다.');
      return;
    }

    setRoomsState((prev) => prev.map((room) => ({ ...room, extra: Array.isArray(room.extra) ? room.extra : [] })));
    setDoorsState((prev) => prev.map((door) => ({ ...door, extra: Array.isArray(door.extra) ? door.extra : [] })));
    if (!stepOneResult?.id) {
      navigate('/admin/upload');
      return;
    }

    if (isBaseSaving) {
      return;
    }

    setIsBaseSaving(true);
    setStepThreeError(null);
    try {
      await persistStepThreeDraft({ force: true });
      navigate('/admin/upload');
    } catch (error) {
      console.error('Failed to save basic info', error);
      setStepThreeError('기본 정보를 저장하지 못했습니다. 잠시 후 다시 시도해 주세요.');
    } finally {
      setIsBaseSaving(false);
    }
  };

  const handleDetailSave = async () => {
    if (!stepOneResult?.id) {
      return;
    }

    setIsDetailSaving(true);
    setDetailSaveError(null);
    try {
      const record = await persistStepThreeDraft({ force: true });
      setStepThreeRecord(record ?? null);
      navigate('/admin/upload');
    } catch (error) {
      console.error('Failed to save detail info', error);
      setDetailSaveError('상세 정보를 저장하지 못했습니다. 잠시 후 다시 시도해 주세요.');
    } finally {
      setIsDetailSaving(false);
    }
  };

  const handleBackToUpload = () => {
    navigate('/admin/upload');
  };

  const needsRoomHydration = roomEntities.length > 0 && roomsState.length < roomEntities.length;
  const needsDoorHydration = doorEntities.length > 0 && doorsState.length < doorEntities.length;
  const isListHydrationPending = needsRoomHydration || needsDoorHydration;
  const isProcessingPending = isProcessingLoading && !processingData && !processingError;
  const isStepThreePending = !isStepThreeFetchComplete || (isStepThreeLoading && !stepThreeRecord && !stepThreeError);
  const isStoredHydrationPending = Boolean(stepThreeRecord) && !hasHydratedStepThree.current;
  const shouldDelayRendering =
    (isProcessingPending || isStepThreePending || isListHydrationPending || isStoredHydrationPending) &&
    !stepThreeError &&
    !processingError;

  if (!stepOneResult) {
    return null;
  }

  if (shouldDelayRendering) {
    return (
      <div ref={pageRef} className={styles.page} style={pageStyle}>
        <div className={styles.loadingWrapper}>
          <div className={styles.loadingCard}>3단계 저장본을 불러오는 중입니다...</div>
        </div>
      </div>
    );
  }

  return (
    <div ref={pageRef} className={styles.page} style={pageStyle}>
      <header className={styles.header}>
        <button type='button' className={styles.backButton} onClick={handleBackToUpload}>
          <ArrowLeft size={18} />
          메인으로
        </button>
        <div className={styles.heading}>
          <h1>3단계: 그래프 노드 메타데이터 입력</h1>
          <p>도면 ID: {stepOneResult.fileName ?? stepOneResult.id}</p>
          {stepOneFloorLabel && <p>층 정보: {stepOneFloorLabel}</p>}
          <p>그래프 ID: {processingRequestId ?? '생성되지 않음'}</p>
        </div>
        <div className={styles.stageIndicator}>
          <span className={stage === 'base' ? styles.activeStage : ''}>기본 정보</span>
          <span className={stage === 'details' ? styles.activeStage : ''}>상세 정보</span>
        </div>
        <div className={styles.toggleGroup}>
          <button
            type='button'
            className={`${styles.toggleButton} ${showRoomLabels ? styles.toggleButtonActive : ''}`}
            onClick={() => setShowRoomLabels((prev) => !prev)}
          >
            방 ID {showRoomLabels ? '숨기기' : '표시'}
          </button>
          <button
            type='button'
            className={`${styles.toggleButton} ${showDoorLabels ? styles.toggleButtonActive : ''}`}
            onClick={() => setShowDoorLabels((prev) => !prev)}
          >
            문 ID {showDoorLabels ? '숨기기' : '표시'}
          </button>
        </div>
      </header>

      <div className={styles.workspace}>
        <div className={styles.canvasColumn}>
          <StepTwoCanvas
            imageSize={imageSize}
            graph={processingGraph}
            rooms={roomEntities}
            doors={doorEntities}
            wallLines={wallLineEntities}
            stairs={stairEntities}
            elevators={elevatorEntities}
            selectedEntity={selectedEntity}
            onSelectEntity={handleSelectEntity}
            isLoading={isProcessingLoading}
            error={processingError}
            showRoomLabels={showRoomLabels}
            showDoorLabels={showDoorLabels}
          />
        </div>

        <div className={styles.formColumn}>
          {stage === 'base' && (
            <div className={styles.formContent}>
              {processingError && <div className={styles.noticeBox}>{processingError}</div>}
              {isStepThreeLoading && <div className={styles.noticeBox}>이전 3단계 데이터를 불러오는 중...</div>}
              {stepThreeError && <p className={styles.errorMessage}>{stepThreeError}</p>}
              <section className={`${styles.section} ${styles.sectionGrow}`}>
                <div className={styles.sectionHeader}>
                  <div>
                    <h2>Room 노드 기본 정보</h2>
                    <p>Room 박스 {roomEntities.length}개의 이름과 호수를 입력하세요.</p>
                  </div>
                </div>
                <div className={styles.sectionScroll}>
                  <div className={styles.entityList}>
                    {roomEntities.length === 0 && <p className={styles.emptyMessage}>등록된 Room 박스가 없습니다.</p>}
                    {roomsState.map((room) => {
                      const entity = roomEntityMap.get(room.nodeId);
                      const isActive = selectedEntity?.type === 'room' && selectedEntity.nodeId === room.nodeId;
                      return (
                        <div
                          key={room.nodeId}
                          ref={(element) => registerRoomListRef(room.nodeId, element)}
                          className={`${styles.entityRow} ${isActive ? styles.entityRowActive : ''}`}
                          onMouseEnter={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                          onClick={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                        >
                          <div className={styles.nodeBadge}>
                            <strong>{entity?.index ?? '-'}</strong>
                            <span>{entity?.label ?? room.nodeId}</span>
                          </div>
                          <div className={styles.baseFields}>
                            <label>
                              <span>방 이름</span>
                              <input
                                type='text'
                                placeholder='예: 터만홀'
                                value={room.name}
                                onFocus={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                                onChange={(event) => handleRoomFieldChange(room.nodeId, 'name', event.target.value)}
                              />
                            </label>
                            <label>
                              <span>방 호수</span>
                              <input
                                type='text'
                                placeholder='예: 301'
                                value={room.number}
                                onFocus={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                                onChange={(event) => handleRoomFieldChange(room.nodeId, 'number', event.target.value)}
                              />
                            </label>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </section>

              <section className={`${styles.section} ${styles.sectionGrow}`}>
                <div className={styles.sectionHeader}>
                  <div>
                    <h2>Door 노드 기본 정보</h2>
                    <p>문 종류를 선택하고 필요하면 직접 입력해 주세요.</p>
                  </div>
                </div>
                <div className={styles.sectionScroll}>
                  <div className={styles.entityList}>
                    {doorEntities.length === 0 && <p className={styles.emptyMessage}>등록된 Door 포인트가 없습니다.</p>}
                    {doorsState.map((door) => {
                      const entity = doorEntityMap.get(door.nodeId);
                      const isActive = selectedEntity?.type === 'door' && selectedEntity.nodeId === door.nodeId;
                      return (
                        <div
                          key={door.nodeId}
                          ref={(element) => registerDoorListRef(door.nodeId, element)}
                          className={`${styles.entityRow} ${isActive ? styles.entityRowActive : ''}`}
                          onMouseEnter={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                          onClick={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                        >
                          <div className={styles.nodeBadge}>
                            <strong>{entity?.index ?? '-'}</strong>
                            <span>{entity?.label ?? door.nodeId}</span>
                          </div>
                          <div className={styles.baseFields}>
                            <label>
                              <span>문 종류</span>
                              <select
                                value={door.type}
                                onFocus={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                                onChange={(event) => handleDoorFieldChange(door.nodeId, 'type', event.target.value)}
                              >
                                <option value=''>선택하세요</option>
                                <option value='미닫이'>미닫이</option>
                                <option value='여닫이'>여닫이</option>
                                <option value='기타'>기타</option>
                              </select>
                            </label>
                            {door.type === '기타' && (
                              <label>
                                <span>직접 입력</span>
                                <input
                                  type='text'
                                  placeholder='예: 양개문'
                                  value={door.customType}
                                  onFocus={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                                  onChange={(event) =>
                                    handleDoorFieldChange(door.nodeId, 'customType', event.target.value)
                                  }
                                />
                              </label>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </section>

              <footer className={styles.footer}>
                <button
                  type='button'
                  className={styles.primaryButton}
                  onClick={handleBaseSave}
                  disabled={isStepThreeLoading || isBaseSaving}
                >
                  {isBaseSaving ? '저장 중...' : isStepThreeLoading ? '불러오는 중...' : '저장하기'}
                  <ArrowRight size={16} />
                </button>
              </footer>
            </div>
          )}

          {stage === 'details' && (
            <div className={styles.formContent}>
              {processingError && <div className={styles.noticeBox}>{processingError}</div>}
              <section className={`${styles.section} ${styles.sectionGrow}`}>
                <div className={styles.sectionHeader}>
                  <div>
                    <h2>Room 상세 정보</h2>
                    <p>기본 정보를 확인하고 필요한 추가 필드를 입력하세요.</p>
                  </div>
                </div>
                <div className={styles.detailsScroll}>
                  <div className={styles.detailsList}>
                    {roomsState.map((room) => {
                      const displayLabel = buildRoomDisplayLabel(room.name, room.number);
                      const title = displayLabel || room.name || room.nodeId;
                      const subtitle = displayLabel ? room.nodeId : '기본 정보 미입력';
                      const isActive = selectedEntity?.type === 'room' && selectedEntity.nodeId === room.nodeId;
                      return (
                        <details
                          key={room.nodeId}
                          ref={(element) => registerRoomDetailsRef(room.nodeId, element)}
                          data-room-id={room.nodeId}
                          className={`${styles.detailsItem} ${isActive ? styles.entityRowActive : ''}`}
                          onMouseEnter={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                        >
                          <summary
                            className={styles.entitySummary}
                            onClick={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                          >
                            <span>{title}</span>
                            <span className={styles.summaryNote}>{subtitle}</span>
                          </summary>
                          <div className={styles.detailBody}>
                            <KeyValueEditor
                              entries={room.extra}
                              onChange={(extra) => handleRoomExtraChange(room.nodeId, extra)}
                              addButtonLabel='추가 정보 필드'
                            />
                          </div>
                        </details>
                      );
                    })}
                  </div>
                </div>
              </section>

              <section className={`${styles.section} ${styles.sectionGrow}`}>
                <div className={styles.sectionHeader}>
                  <div>
                    <h2>Door 상세 정보</h2>
                    <p>문 종류를 확인하고 추가 정보를 입력하세요.</p>
                  </div>
                </div>
                <div className={styles.detailsScroll}>
                  <div className={styles.detailsList}>
                    {doorsState.map((door) => {
                      const resolvedType = door.type === '기타' ? door.customType : door.type;
                      const title = resolvedType || '문 정보 미입력';
                      const subtitle = door.type ? door.nodeId : '문 종류를 선택하세요';
                      const isActive = selectedEntity?.type === 'door' && selectedEntity.nodeId === door.nodeId;
                      return (
                        <details
                          key={door.nodeId}
                          ref={(element) => registerDoorDetailsRef(door.nodeId, element)}
                          data-door-id={door.nodeId}
                          className={`${styles.detailsItem} ${isActive ? styles.entityRowActive : ''}`}
                          onMouseEnter={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                        >
                          <summary
                            className={styles.entitySummary}
                            onClick={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                          >
                            <span>{title}</span>
                            <span className={styles.summaryNote}>{subtitle}</span>
                          </summary>
                          <div className={styles.detailBody}>
                            <KeyValueEditor
                              entries={door.extra}
                              onChange={(extra) => handleDoorExtraChange(door.nodeId, extra)}
                              addButtonLabel='추가 정보 필드'
                            />
                          </div>
                        </details>
                      );
                    })}
                  </div>
                </div>
              </section>

              {detailSaveError && <p className={styles.errorMessage}>{detailSaveError}</p>}

              <footer className={styles.footer}>
                <button
                  type='button'
                  className={styles.primaryButton}
                  onClick={handleDetailSave}
                  disabled={isDetailSaving || isStepThreeLoading}
                >
                  {isDetailSaving ? '저장 중...' : '저장하기'}
                  <Save size={16} />
                </button>
              </footer>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdminStepThreePage;
