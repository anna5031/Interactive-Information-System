import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, RefreshCw, Save, X } from 'lucide-react';
import StepTwoCanvas from '../components/stepTwo/StepTwoCanvas';
import {
  fetchFloorPlanFloors,
  fetchFloorPlanGraph,
  fetchProcessingResultById,
  fetchStoredFloorPlanSummaries,
  saveFloorPlanGraph,
} from '../api/floorPlans';
import { saveStepOneResult } from '../api/stepOneResults';
import styles from './AdminGraphEditorPage.module.css';

const enhanceGraphData = (graph) => {
  const safeGraph = graph || {};
  const nodes = Array.isArray(safeGraph.nodes) ? safeGraph.nodes : [];
  const edges = Array.isArray(safeGraph.edges) ? safeGraph.edges : [];
  return {
    nodes: nodes.map((node, index) => {
      const pos = Array.isArray(node?.pos) ? node.pos : [];
      return {
        id: node?.id ?? `node_${index}`,
        type: node?.type ?? '',
        pos,
        posText: pos.length ? pos.join(', ') : '',
        attributes: node && typeof node.attributes === 'object' && node.attributes !== null ? node.attributes : {},
      };
    }),
    edges: edges.map((edge) => ({
      source: edge?.source ?? '',
      target: edge?.target ?? '',
      weight: Number.isFinite(edge?.weight) ? edge.weight : undefined,
      weightText: Number.isFinite(edge?.weight) ? String(edge.weight) : '',
      attributes: edge && typeof edge.attributes === 'object' && edge.attributes !== null ? edge.attributes : {},
    })),
  };
};

const buildGraphPayload = (graph) => {
  const safeGraph = graph || { nodes: [], edges: [] };
  const nodes = Array.isArray(safeGraph.nodes)
    ? safeGraph.nodes.map((node) => {
        const sanitized = { ...node };
        delete sanitized.posText;
        return sanitized;
      })
    : [];
  const edges = Array.isArray(safeGraph.edges)
    ? safeGraph.edges.map((edge) => {
        const sanitized = { ...edge };
        delete sanitized.weightText;
        return sanitized;
      })
    : [];
  return { nodes, edges };
};

const toPolygonPoints = (polygonLike) => {
  if (Array.isArray(polygonLike)) {
    return polygonLike;
  }
  if (
    polygonLike &&
    typeof polygonLike === 'object' &&
    Array.isArray(polygonLike.coordinates) &&
    Array.isArray(polygonLike.coordinates[0])
  ) {
    return polygonLike.coordinates[0];
  }
  return null;
};

const toPolygonString = (points) => {
  if (!Array.isArray(points) || points.length === 0) {
    return null;
  }
  return points.map((point) => point.map((value) => Number(value).toFixed(1)).join(',')).join(' ');
};

const toPointArray = (value) => {
  if (Array.isArray(value) && value.length >= 2) {
    return [Number(value[0]), Number(value[1])];
  }
  if (value && typeof value === 'object' && Array.isArray(value.coordinates)) {
    return toPointArray(value.coordinates);
  }
  return null;
};

const AdminGraphEditorPage = () => {
  const { requestId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const locationStepOneResult = location.state?.stepOneResult ?? null;
  const locationStepOneId = location.state?.stepOneId ?? locationStepOneResult?.id ?? null;

  const [graphState, setGraphState] = useState({ nodes: [], edges: [] });
  const [graphDirty, setGraphDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [objects, setObjects] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 1000, height: 1000 });
  const [metadata, setMetadata] = useState(null);
  const [processingResult, setProcessingResult] = useState(locationStepOneResult?.processingResult ?? null);

  const graphPayload = useMemo(() => buildGraphPayload(graphState), [graphState]);
  const graphNodeIds = useMemo(() => graphState.nodes.map((node) => node.id).filter(Boolean), [graphState.nodes]);
  const [graphSelection, setGraphSelection] = useState([]);
  const [isCrossFloorModalOpen, setIsCrossFloorModalOpen] = useState(false);
  const [crossFloorType, setCrossFloorType] = useState('stair');
  const [selectedLocalConnectorId, setSelectedLocalConnectorId] = useState(null);
  const [floorOptions, setFloorOptions] = useState([]);
  const [hasRequestedFloorOptions, setHasRequestedFloorOptions] = useState(false);
  const [isLoadingFloorOptions, setIsLoadingFloorOptions] = useState(false);
  const [floorSummaryMap, setFloorSummaryMap] = useState(() => new Map());
  const [isLoadingFloorSummaries, setIsLoadingFloorSummaries] = useState(false);
  const [floorSummaryError, setFloorSummaryError] = useState(null);
  const [hasLoadedFloorSummaries, setHasLoadedFloorSummaries] = useState(false);
  const [selectedTargetFloorId, setSelectedTargetFloorId] = useState(null);
  const [remoteConnectorData, setRemoteConnectorData] = useState({ requestId: null, stair: [], elevator: [] });
  const [remoteGraphCache, setRemoteGraphCache] = useState({});
  const [isRemoteConnectorLoading, setIsRemoteConnectorLoading] = useState(false);
  const [remoteConnectorError, setRemoteConnectorError] = useState(null);
  const [selectedRemoteConnectorId, setSelectedRemoteConnectorId] = useState(null);
  const [crossFloorModalError, setCrossFloorModalError] = useState(null);
  const currentFloorLabel = useMemo(() => {
    const metadataFloor =
      metadata?.floor_label ??
      metadata?.floorLabel ??
      locationStepOneResult?.metadata?.floorLabel ??
      locationStepOneResult?.metadata?.floor_label ??
      null;
    if (metadataFloor) {
      return metadataFloor;
    }
    if (locationStepOneResult?.floorLabel) {
      return locationStepOneResult.floorLabel;
    }
    if (!requestId) {
      return null;
    }
    const summary = floorSummaryMap.get(String(requestId));
    if (!summary) {
      return null;
    }
    return summary.floorLabel ?? summary.floorValue ?? summary.label ?? null;
  }, [floorSummaryMap, locationStepOneResult, metadata, requestId]);

  const doorEndpointMap = useMemo(() => {
    const map = new Map();
    (graphState.nodes ?? []).forEach((node) => {
      if (node?.type !== 'door_endpoints') {
        return;
      }
      const links = node?.attributes?.door_link_ids || node?.attributes?.doorLinkIds || [];
      links.forEach((linkId) => {
        const doorId =
          typeof linkId === 'number'
            ? `door_${linkId}`
            : String(linkId).startsWith('door_')
              ? String(linkId)
              : `door_${linkId}`;
        if (!map.has(doorId)) {
          map.set(doorId, new Set());
        }
        map.get(doorId).add(node.id);
      });
    });
    return map;
  }, [graphState.nodes]);

  const localConnectorNodes = useMemo(() => {
    const nodes = graphState.nodes ?? [];
    const build = (type) =>
      nodes
        .filter((node) => (node?.type || '').toLowerCase() === type)
        .map((node, index) => ({
          id: node?.id ?? `${type}_${index}`,
          label: node?.id ?? `${type}_${index}`,
        }));
    return {
      stair: build('stair'),
      elevator: build('elevator'),
    };
  }, [graphState.nodes]);

  const refreshProcessingResult = useCallback(async () => {
    if (!requestId) {
      return null;
    }
    try {
      const response = await fetchProcessingResultById(requestId);
      setProcessingResult(response);
      if (response?.image_size) {
        setImageSize({
          width: Number(response.image_size.width) || 1000,
          height: Number(response.image_size.height) || 1000,
        });
      }
      return response;
    } catch (fetchError) {
      console.error('Failed to fetch processing result', fetchError);
      return null;
    }
  }, [requestId]);

  const loadProcessingResult = useCallback(async () => {
    if (processingResult || !requestId) {
      return;
    }
    await refreshProcessingResult();
  }, [processingResult, refreshProcessingResult, requestId]);

  const loadFloorSummaries = useCallback(async () => {
    setIsLoadingFloorSummaries(true);
    setFloorSummaryError(null);
    setHasLoadedFloorSummaries(false);
    try {
      const summaries = await fetchFloorPlanFloors();
      const normalized = new Map();
      (summaries ?? []).forEach((entry) => {
        const rawRequestId =
          entry?.request_id ?? entry?.requestId ?? entry?.id ?? entry?.metadata?.requestId ?? null;
        if (!rawRequestId) {
          return;
        }
        const requestKey = String(rawRequestId);
        const floorLabel = entry?.floor_label ?? entry?.floorLabel ?? null;
        const floorValue = entry?.floor_value ?? entry?.floorValue ?? null;
        const label = floorLabel || floorValue || `요청 ${requestKey}`;
        normalized.set(requestKey, {
          floorLabel,
          floorValue,
          label,
        });
      });
      setFloorSummaryMap(normalized);
      setHasLoadedFloorSummaries(true);
    } catch (summaryError) {
      console.error('Failed to load floor summaries', summaryError);
      setFloorSummaryMap(new Map());
      setFloorSummaryError('층 정보를 불러오지 못했습니다. 요청 ID만 표시됩니다.');
    } finally {
      setIsLoadingFloorSummaries(false);
    }
  }, []);

  const loadGraph = useCallback(async () => {
    if (!requestId) {
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const payload = await fetchFloorPlanGraph(requestId);
      if (payload?.graph) {
        setGraphState(enhanceGraphData(payload.graph));
      } else {
        setGraphState({ nodes: [], edges: [] });
      }
      setObjects(payload?.objects ?? null);
      setMetadata(payload?.metadata ?? null);
      const metaSize = payload?.metadata?.image_size ?? payload?.metadata?.imageSize;
      if (metaSize?.width && metaSize?.height) {
        setImageSize({
          width: Number(metaSize.width) || 1000,
          height: Number(metaSize.height) || 1000,
        });
      }
      setGraphDirty(false);
    } catch (loadError) {
      console.error('Failed to load graph data', loadError);
      setError('그래프 데이터를 불러오지 못했습니다.');
    } finally {
      setIsLoading(false);
    }
  }, [requestId]);

  useEffect(() => {
    loadGraph();
    loadProcessingResult();
  }, [loadGraph, loadProcessingResult]);

  const loadFloorOptions = useCallback(async () => {
    if (!requestId) {
      setHasRequestedFloorOptions(true);
      return;
    }
    setIsLoadingFloorOptions(true);
    try {
      const summaries = await fetchStoredFloorPlanSummaries();
      const options =
        summaries
          ?.map((item, index) => {
            const summaryRequestId =
              item?.request_id ??
              item?.requestId ??
              item?.id ??
              item?.metadata?.request_id ??
              item?.metadata?.requestId ??
              null;
            if (!summaryRequestId || String(summaryRequestId) === String(requestId)) {
              return null;
            }
            const label =
              item?.floor_label ??
              item?.floorLabel ??
              item?.metadata?.floor_label ??
              item?.metadata?.floorLabel ??
              item?.metadata?.file_name ??
              item?.metadata?.fileName ??
              item?.file_name ??
              item?.fileName ??
              item?.name ??
              `요청 ${summaryRequestId}` ??
              `요청 ${index + 1}`;
            const fileNameLabel =
              item?.metadata?.file_name ??
              item?.metadata?.fileName ??
              item?.file_name ??
              item?.fileName ??
              item?.name ??
              `요청 ${summaryRequestId}` ??
              `요청 ${index + 1}`;
            return {
              requestId: String(summaryRequestId),
              label,
              floorLabel:
                item?.floor_label ??
                item?.floorLabel ??
                item?.metadata?.floor_label ??
                item?.metadata?.floorLabel ??
                null,
              fileName: fileNameLabel,
            };
          })
          .filter(Boolean) ?? [];
      setFloorOptions(options);
    } catch (error) {
      console.error('Failed to load floor options', error);
      setFloorOptions([]);
    } finally {
      setIsLoadingFloorOptions(false);
      setHasRequestedFloorOptions(true);
    }
  }, [requestId]);

  const ensureRemoteGraph = useCallback(
    async (targetFloorId) => {
      const key = String(targetFloorId);
      if (remoteGraphCache[key]) {
        return remoteGraphCache[key];
      }
      const payload = await fetchFloorPlanGraph(targetFloorId);
      const remoteGraph = payload?.graph ? enhanceGraphData(payload.graph) : { nodes: [], edges: [] };
      setRemoteGraphCache((prev) => ({
        ...prev,
        [key]: remoteGraph,
      }));
      return remoteGraph;
    },
    [remoteGraphCache]
  );

  const loadRemoteConnectorNodes = useCallback(
    async (targetFloorId) => {
      if (!targetFloorId) {
        return;
      }
      setIsRemoteConnectorLoading(true);
      setRemoteConnectorError(null);
      setCrossFloorModalError(null);
      setSelectedRemoteConnectorId(null);
      try {
        const remoteGraph = await ensureRemoteGraph(targetFloorId);
        const nodes = remoteGraph?.nodes ?? [];
        const buildConnectorList = (type) =>
          nodes
            .filter((node) => (node?.type || '').toLowerCase() === type)
            .map((node, index) => ({
              id: node?.id ?? `${type}_${index}`,
              label: node?.id ?? `${type}_${index}`,
            }));
        setRemoteConnectorData({
          requestId: String(targetFloorId),
          stair: buildConnectorList('stair'),
          elevator: buildConnectorList('elevator'),
        });
        setRemoteGraphCache((prev) => ({
          ...prev,
          [String(targetFloorId)]: remoteGraph,
        }));
      } catch (error) {
        console.error('Failed to load remote connector nodes', error);
        setRemoteConnectorData({ requestId: String(targetFloorId), stair: [], elevator: [] });
        setRemoteConnectorError('다른 도면 정보를 불러오지 못했습니다.');
      } finally {
        setIsRemoteConnectorLoading(false);
      }
    },
    [ensureRemoteGraph]
  );

  useEffect(() => {
    if (!isCrossFloorModalOpen) {
      setHasRequestedFloorOptions(false);
      return;
    }
    if (hasRequestedFloorOptions || isLoadingFloorOptions) {
      return;
    }
    loadFloorOptions();
  }, [hasRequestedFloorOptions, isCrossFloorModalOpen, isLoadingFloorOptions, loadFloorOptions]);

  useEffect(() => {
    if (processingResult?.objects && !objects) {
      setObjects(processingResult.objects);
    }
  }, [processingResult, objects]);

  useEffect(() => {
    loadFloorSummaries();
  }, [loadFloorSummaries]);

  useEffect(() => {
    setGraphSelection((prev) => prev.filter((id) => graphNodeIds.includes(id)));
  }, [graphNodeIds]);

  const roomEntities = useMemo(() => {
    if (!objects?.rooms) {
      return [];
    }
    return objects.rooms
      .map((room, index) => {
        const polygonPoints = toPolygonPoints(room?.polygon) || (Array.isArray(room?.corners) ? room.corners : null);
        const polygon = toPolygonString(polygonPoints);
        const centroid = toPointArray(room?.centroid);
        return {
          nodeId: room?.source_id ? String(room.source_id) : `room_${room?.id ?? index}`,
          label: room?.properties?.name ?? room?.label ?? `Room ${index + 1}`,
          polygon,
          labelPosition: centroid,
        };
      })
      .filter(Boolean);
  }, [objects]);

  const doorObjectsBySourceId = useMemo(() => {
    const map = new Map();
    (objects?.doors ?? []).forEach((door, index) => {
      const sourceId = door?.source_id ?? door?.id ?? `door_object_${index}`;
      map.set(String(sourceId), door);
    });
    return map;
  }, [objects]);

  const stairObjectsBySourceId = useMemo(() => {
    const map = new Map();
    (objects?.stairs ?? []).forEach((stair, index) => {
      const sourceId = stair?.source_id ?? stair?.id ?? `stair_object_${index}`;
      map.set(String(sourceId), stair);
    });
    return map;
  }, [objects]);

  const elevatorObjectsBySourceId = useMemo(() => {
    const map = new Map();
    (objects?.elevators ?? []).forEach((elevator, index) => {
      const sourceId = elevator?.source_id ?? elevator?.id ?? `elevator_object_${index}`;
      map.set(String(sourceId), elevator);
    });
    return map;
  }, [objects]);

  const doorEntities = useMemo(() => {
    const graphDoorNodes = (graphState.nodes ?? []).filter((node) => node.type === 'door');
    const sortedDoorNodes = graphDoorNodes.slice().sort((a, b) => {
      const aId = Number.parseInt(String(a.id).replace('door_', ''), 10);
      const bId = Number.parseInt(String(b.id).replace('door_', ''), 10);
      if (!Number.isNaN(aId) && !Number.isNaN(bId)) {
        return aId - bId;
      }
      return String(a.id).localeCompare(String(b.id));
    });

    if (sortedDoorNodes.length > 0) {
      return sortedDoorNodes
        .map((node, index) => {
          const attrs = node?.attributes ?? {};
          const sourceId = attrs?.source_id ?? attrs?.sourceId ?? attrs?.door_source_id ?? attrs?.doorSourceId ?? null;
          const doorObject =
            (sourceId && doorObjectsBySourceId.get(String(sourceId))) || (objects?.doors ?? [])[index] || null;
          const polygonPoints =
            toPolygonPoints(doorObject?.polygon) || (Array.isArray(doorObject?.corners) ? doorObject.corners : null);
          let centroid = toPointArray(doorObject?.centroid);
          if ((!centroid || centroid.some((value) => !Number.isFinite(value))) && Array.isArray(node.pos)) {
            centroid = [Number(node.pos[0]), Number(node.pos[1])];
          }
          let polygon = toPolygonString(polygonPoints);
          if (!polygon && centroid) {
            const [cx, cy] = centroid;
            const size = 6;
            polygon = toPolygonString([
              [cx - size, cy - size],
              [cx + size, cy - size],
              [cx + size, cy + size],
              [cx - size, cy + size],
            ]);
          }
          return {
            nodeId: node.id,
            index: index + 1,
            label: doorObject?.properties?.name ?? doorObject?.label ?? node.id,
            polygon,
            centroid,
          };
        })
        .filter(Boolean);
    }

    const fallbackDoors = (objects?.doors ?? []).map((door, index) => {
      const polygonPoints = toPolygonPoints(door?.polygon) || (Array.isArray(door?.corners) ? door.corners : null);
      let centroid = toPointArray(door?.centroid);
      if ((!centroid || centroid.some((value) => !Number.isFinite(value))) && Array.isArray(door?.centroid)) {
        centroid = toPointArray(door.centroid);
      }
      let polygon = toPolygonString(polygonPoints);
      if (!polygon && centroid) {
        const [cx, cy] = centroid;
        const size = 6;
        polygon = toPolygonString([
          [cx - size, cy - size],
          [cx + size, cy - size],
          [cx + size, cy + size],
          [cx - size, cy + size],
        ]);
      }
      return {
        nodeId: door?.source_id ? String(door.source_id) : `door_${door?.id ?? index}`,
        index: index + 1,
        label: door?.properties?.name ?? door?.label ?? `Door ${index + 1}`,
        polygon,
        centroid,
      };
    });
    return fallbackDoors.filter(Boolean);
  }, [graphState.nodes, doorObjectsBySourceId, objects]);

  const wallLineEntities = useMemo(() => {
    const segments = Array.isArray(objects?.wall_segments) ? objects.wall_segments : [];
    return segments
      .map((segment, index) => {
        const start = Array.isArray(segment?.start) ? segment.start : null;
        const end = Array.isArray(segment?.end) ? segment.end : null;
        if (!start || !end) {
          return null;
        }
        return {
          id: segment?.id ?? `wall_${index}`,
          x1: Number(start[0]),
          y1: Number(start[1]),
          x2: Number(end[0]),
          y2: Number(end[1]),
        };
      })
      .filter(Boolean);
  }, [objects]);

  const stairEntities = useMemo(() => {
    const graphStairs = (graphState.nodes ?? []).filter((node) => (node?.type || '').toLowerCase() === 'stair');
    if (graphStairs.length > 0) {
      return graphStairs
        .map((node, index) => {
          const stairObject =
            stairObjectsBySourceId.get(node?.attributes?.source_id ?? node?.attributes?.sourceId) ||
            (objects?.stairs ?? [])[index] ||
            null;
          const polygonPoints =
            toPolygonPoints(stairObject?.polygon) || (Array.isArray(stairObject?.corners) ? stairObject.corners : null);
          let centroid = toPointArray(stairObject?.centroid);
          if ((!centroid || centroid.some((value) => !Number.isFinite(value))) && Array.isArray(node.pos)) {
            centroid = [Number(node.pos[0]), Number(node.pos[1])];
          }
          let polygon = toPolygonString(polygonPoints);
          if (!polygon && centroid) {
            const [cx, cy] = centroid;
            const size = 6;
            polygon = toPolygonString([
              [cx - size, cy - size],
              [cx + size, cy - size],
              [cx + size, cy + size],
              [cx - size, cy + size],
            ]);
          }
          if (!polygon) {
            return null;
          }
          return {
            id: node.id,
            nodeId: node.id,
            polygon,
            centroid,
            label: stairObject?.properties?.name ?? stairObject?.label ?? node.id,
          };
        })
        .filter(Boolean);
    }

    return (objects?.stairs ?? [])
      .map((stair, index) => {
        let polygonPoints = toPolygonPoints(stair?.polygon) || (Array.isArray(stair?.corners) ? stair.corners : null);
        let centroid = toPointArray(stair?.centroid);
        if (!polygonPoints && centroid) {
          const [cx, cy] = centroid;
          const size = 6;
          polygonPoints = [
            [cx - size, cy - size],
            [cx + size, cy - size],
            [cx + size, cy + size],
            [cx - size, cy + size],
          ];
        }
        if (!polygonPoints || polygonPoints.length < 3) {
          return null;
        }
        return {
          id: stair?.source_id ? String(stair.source_id) : `stair_${stair?.id ?? index}`,
          nodeId: stair?.source_id ? String(stair.source_id) : `stair_${stair?.id ?? index}`,
          polygon: toPolygonString(polygonPoints),
          centroid,
          label: stair?.properties?.name ?? stair?.label ?? `stair ${index + 1}`,
        };
      })
      .filter(Boolean);
  }, [graphState.nodes, stairObjectsBySourceId, objects]);

  const elevatorEntities = useMemo(() => {
    const graphElevators = (graphState.nodes ?? []).filter((node) => (node?.type || '').toLowerCase() === 'elevator');
    if (graphElevators.length > 0) {
      return graphElevators
        .map((node, index) => {
          const elevatorObject =
            elevatorObjectsBySourceId.get(node?.attributes?.source_id ?? node?.attributes?.sourceId) ||
            (objects?.elevators ?? [])[index] ||
            null;
          const polygonPoints =
            toPolygonPoints(elevatorObject?.polygon) ||
            (Array.isArray(elevatorObject?.corners) ? elevatorObject.corners : null);
          let centroid = toPointArray(elevatorObject?.centroid);
          if ((!centroid || centroid.some((value) => !Number.isFinite(value))) && Array.isArray(node.pos)) {
            centroid = [Number(node.pos[0]), Number(node.pos[1])];
          }
          let polygon = toPolygonString(polygonPoints);
          if (!polygon && centroid) {
            const [cx, cy] = centroid;
            const size = 6;
            polygon = toPolygonString([
              [cx - size, cy - size],
              [cx + size, cy - size],
              [cx + size, cy + size],
              [cx - size, cy + size],
            ]);
          }
          if (!polygon) {
            return null;
          }
          return {
            id: node.id,
            nodeId: node.id,
            polygon,
            centroid,
            label: elevatorObject?.properties?.name ?? elevatorObject?.label ?? node.id,
          };
        })
        .filter(Boolean);
    }

    return (objects?.elevators ?? [])
      .map((el, index) => {
        let polygonPoints = toPolygonPoints(el?.polygon) || (Array.isArray(el?.corners) ? el.corners : null);
        let centroid = toPointArray(el?.centroid);
        if (!polygonPoints && centroid) {
          const [cx, cy] = centroid;
          const size = 6;
          polygonPoints = [
            [cx - size, cy - size],
            [cx + size, cy - size],
            [cx + size, cy + size],
            [cx - size, cy + size],
          ];
        }
        if (!polygonPoints || polygonPoints.length < 3) {
          return null;
        }
        return {
          id: el?.source_id ? String(el.source_id) : `elevator_${el?.id ?? index}`,
          nodeId: el?.source_id ? String(el.source_id) : `elevator_${el?.id ?? index}`,
          polygon: toPolygonString(polygonPoints),
          centroid,
          label: el?.properties?.name ?? el?.label ?? `elevator ${index + 1}`,
        };
      })
      .filter(Boolean);
  }, [graphState.nodes, elevatorObjectsBySourceId, objects]);

  const handleBack = () => {
    navigate('/admin/upload');
  };

  const handleSave = async () => {
    if (!requestId) {
      return;
    }
    setIsSaving(true);
    setError(null);
    try {
      const payload = buildGraphPayload(graphState);
      const response = await saveFloorPlanGraph(requestId, payload);
      if (response?.graph) {
        setGraphState(enhanceGraphData(response.graph));
      }
      setMetadata(response?.metadata ?? null);
      setGraphDirty(false);

      const updatedProcessing = await refreshProcessingResult();
      if (updatedProcessing && locationStepOneId) {
        try {
          await saveStepOneResult({
            sourceOriginalId: locationStepOneId,
            requestId: updatedProcessing?.request_id ?? requestId,
            processingResult: updatedProcessing,
            metadata: {
              ...(locationStepOneResult?.metadata || {}),
              ...(updatedProcessing?.metadata || {}),
            },
            objectDetection: locationStepOneResult?.objectDetection,
            wall: locationStepOneResult?.wall,
            door: locationStepOneResult?.door,
            preview: updatedProcessing?.metadata?.preview ?? locationStepOneResult?.preview ?? null,
          });
        } catch (persistError) {
          console.error('Failed to persist updated step one record', persistError);
        }
      }

      navigate('/admin/upload');
    } catch (saveError) {
      console.error('Failed to save graph', saveError);
      setError('그래프 저장에 실패했습니다. 입력 값을 확인해 주세요.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleNodePositionChange = useCallback((nodeId, position) => {
    setGraphState((prev) => ({
      ...prev,
      nodes: prev.nodes.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              pos: position,
              posText: position.map((value) => value.toFixed(2)).join(', '),
            }
          : node
      ),
    }));
    setGraphDirty(true);
  }, []);

  const handleAddNode = useCallback(
    (type = 'corridor') => {
      const nodes = Array.isArray(graphState.nodes) ? graphState.nodes : [];
      const prefix = `${type}_`;
      const maxIndex = nodes.reduce((max, node) => {
        if (typeof node?.id === 'string' && node.id.startsWith(prefix)) {
          const parsed = Number.parseInt(node.id.slice(prefix.length), 10);
          if (!Number.isNaN(parsed)) {
            return Math.max(max, parsed);
          }
        }
        return max;
      }, -1);
      const newId = `${prefix}${maxIndex + 1}`;
      const pos = [
        Math.round(Math.max(imageSize?.width ?? 1000, 0) / 2),
        Math.round(Math.max(imageSize?.height ?? 1000, 0) / 2),
      ];
      setGraphState((prev) => ({
        ...prev,
        nodes: [
          ...(prev.nodes ?? []),
          {
            id: newId,
            type,
            pos,
            posText: pos.map((value) => value.toFixed(2)).join(', '),
            attributes: {},
          },
        ],
      }));
      setGraphSelection([newId]);
      setGraphDirty(true);
    },
    [graphState.nodes, imageSize]
  );

  const handleOpenCrossFloorModal = () => {
    setIsCrossFloorModalOpen(true);
    setCrossFloorType('stair');
    setSelectedLocalConnectorId(null);
    setSelectedTargetFloorId(null);
    setSelectedRemoteConnectorId(null);
    setRemoteConnectorData({ requestId: null, stair: [], elevator: [] });
    setRemoteConnectorError(null);
    setCrossFloorModalError(null);
  };

  const handleCloseCrossFloorModal = () => {
    setIsCrossFloorModalOpen(false);
    setSelectedLocalConnectorId(null);
    setSelectedTargetFloorId(null);
    setSelectedRemoteConnectorId(null);
    setCrossFloorModalError(null);
    setRemoteConnectorError(null);
  };

  const handleChangeConnectorType = (type) => {
    setCrossFloorType(type);
    setSelectedLocalConnectorId(null);
    setSelectedRemoteConnectorId(null);
    setCrossFloorModalError(null);
  };

  const handleSelectTargetFloor = (targetId) => {
    setSelectedTargetFloorId(targetId || null);
    setSelectedRemoteConnectorId(null);
    setRemoteConnectorError(null);
    setCrossFloorModalError(null);
    if (targetId) {
      loadRemoteConnectorNodes(targetId);
    } else {
      setRemoteConnectorData({ requestId: null, stair: [], elevator: [] });
    }
  };

  const addRemoteCrossFloorEdge = useCallback(
    async ({ remoteRequestId, remoteNodeId, remoteNodeLabel, localNodeId, localNodeLabel, connectorType, isBase }) => {
      if (!remoteRequestId || !remoteNodeId || !requestId) {
        return;
      }
      try {
        const remoteGraph = await ensureRemoteGraph(remoteRequestId);
        const remoteEdges = Array.isArray(remoteGraph?.edges) ? remoteGraph.edges : [];
        const newEdge = {
          source: remoteNodeId,
          target: `cross_${requestId}_${localNodeId}`,
          weight: 1,
          weightText: '1',
          attributes: {
            is_cross_floor: true,
            is_base_cross_floor: Boolean(isBase),
            connector_type: connectorType,
            target_request_id: String(requestId),
            target_node_id: localNodeId,
            target_node_label: localNodeLabel ?? localNodeId,
            source_request_id: String(remoteRequestId),
            source_node_id: remoteNodeId,
            source_node_label: remoteNodeLabel ?? remoteNodeId,
          },
        };
        const filteredEdges = remoteEdges.filter(
          (edge) =>
            !(
              edge?.attributes?.is_cross_floor &&
              edge.source === remoteNodeId &&
              edge?.attributes?.target_request_id === String(requestId)
            )
        );
        const nextGraph = {
          ...remoteGraph,
          edges: [...filteredEdges, newEdge],
        };
        await saveFloorPlanGraph(remoteRequestId, buildGraphPayload(nextGraph));
        setRemoteGraphCache((prev) => ({
          ...prev,
          [String(remoteRequestId)]: nextGraph,
        }));
      } catch (remoteError) {
        console.error('Failed to add remote cross-floor edge', remoteError);
        setError('다른 도면에 연결 정보를 저장하지 못했습니다.');
      }
    },
    [ensureRemoteGraph, requestId]
  );

  const removeRemoteCrossFloorEdge = useCallback(
    async (edge, options = {}) => {
      if (!edge?.attributes?.is_cross_floor) {
        return;
      }
      const remoteRequestId = edge?.attributes?.target_request_id;
      const remoteNodeId = edge?.attributes?.target_node_id;
      if (!remoteRequestId || !remoteNodeId) {
        return;
      }
      try {
        const remoteGraph = await ensureRemoteGraph(remoteRequestId);
        const remoteEdges = Array.isArray(remoteGraph?.edges) ? remoteGraph.edges : [];
        const filteredEdges = remoteEdges.filter(
          (remoteEdge) =>
            !(
              remoteEdge?.attributes?.is_cross_floor &&
              remoteEdge.source === remoteNodeId &&
              remoteEdge?.attributes?.target_request_id === String(requestId) &&
              remoteEdge?.attributes?.target_node_id === edge.source
            )
        );
        if (filteredEdges.length === remoteEdges.length) {
          return;
        }
        const nextGraph = {
          ...remoteGraph,
          edges: filteredEdges,
        };
        await saveFloorPlanGraph(remoteRequestId, buildGraphPayload(nextGraph));
        setRemoteGraphCache((prev) => ({
          ...prev,
          [String(remoteRequestId)]: nextGraph,
        }));
      } catch (remoteError) {
        const isNotFound = remoteError?.response?.status === 404;
        if (isNotFound || options?.silentOnMissing) {
          console.warn('Skipped removing cross-floor edge from missing remote graph', remoteError);
          return;
        }
        console.error('Failed to remove remote cross-floor edge', remoteError);
        setError('다른 도면에서 연결 정보를 삭제하지 못했습니다.');
      }
    },
    [ensureRemoteGraph, requestId]
  );

  const handleCreateCrossFloorConnection = () => {
    if (!selectedLocalConnectorId || !selectedTargetFloorId || !selectedRemoteConnectorId) {
      setCrossFloorModalError('연결할 노드를 모두 선택해주세요.');
      return;
    }
    const remoteOptions =
      remoteConnectorData.requestId === selectedTargetFloorId ? (remoteConnectorData[crossFloorType] ?? []) : [];
    const remoteNode = remoteOptions.find((node) => node.id === selectedRemoteConnectorId) ?? null;
    const localNode = (graphState.nodes ?? []).find((node) => node.id === selectedLocalConnectorId) ?? null;
    const localNodeLabel = localNode?.id ?? selectedLocalConnectorId;
    setGraphState((prev) => ({
      ...prev,
      edges: [
        ...((prev.edges ?? []).filter(
          (edge) =>
            !(
              edge?.attributes?.is_cross_floor &&
              edge.source === selectedLocalConnectorId &&
              edge?.attributes?.target_request_id === selectedTargetFloorId
            )
        )),
        {
          source: selectedLocalConnectorId,
          target: `cross_${selectedTargetFloorId}_${selectedRemoteConnectorId}`,
          weight: 1,
          weightText: '1',
          attributes: {
            is_cross_floor: true,
            is_base_cross_floor: true,
            target_request_id: selectedTargetFloorId,
            target_node_id: selectedRemoteConnectorId,
            target_node_label: remoteNode?.label ?? selectedRemoteConnectorId,
            connector_type: crossFloorType,
            source_request_id: requestId,
            source_node_id: selectedLocalConnectorId,
            source_node_label: localNodeLabel,
          },
        },
      ],
    }));
    setGraphDirty(true);
    handleCloseCrossFloorModal();
    addRemoteCrossFloorEdge({
      remoteRequestId: selectedTargetFloorId,
      remoteNodeId: selectedRemoteConnectorId,
      remoteNodeLabel: remoteNode?.label ?? selectedRemoteConnectorId,
      localNodeId: selectedLocalConnectorId,
      localNodeLabel,
      connectorType: crossFloorType,
      isBase: true,
    });
  };

  const handleRemoveCrossFloorEdge = useCallback(
    (edgeIndex, options = {}) => {
      let removedEdge = null;
      setGraphState((prev) => {
        const edges = prev.edges ?? [];
        if (!edges[edgeIndex]) {
          return prev;
        }
        removedEdge = edges[edgeIndex];
        return {
          ...prev,
          edges: edges.filter((_, index) => index !== edgeIndex),
        };
      });
      if (!removedEdge) {
        return;
      }
      setGraphDirty(true);
      removeRemoteCrossFloorEdge(removedEdge, options);
    },
    [removeRemoteCrossFloorEdge]
  );

  useEffect(() => {
    if (!hasLoadedFloorSummaries) {
      return;
    }
    if (!Array.isArray(graphState.edges) || graphState.edges.length === 0) {
      return;
    }
    const missingTargetIndexes = [];
    (graphState.edges ?? []).forEach((edge, index) => {
      if (!edge?.attributes?.is_cross_floor) {
        return;
      }
      const targetRequestId = edge?.attributes?.target_request_id;
      if (!targetRequestId) {
        missingTargetIndexes.push(index);
        return;
      }
      if (!floorSummaryMap.has(String(targetRequestId))) {
        missingTargetIndexes.push(index);
      }
    });
    if (missingTargetIndexes.length === 0) {
      return;
    }
    missingTargetIndexes
      .sort((a, b) => b - a)
      .forEach((edgeIndex) => handleRemoveCrossFloorEdge(edgeIndex, { silentOnMissing: true }));
  }, [floorSummaryMap, graphState.edges, handleRemoveCrossFloorEdge, hasLoadedFloorSummaries]);

  const handleDeleteSelectedNodes = useCallback(() => {
    if (graphSelection.length === 0) {
      return;
    }
    const corridorIds = new Set(
      (graphState.nodes ?? []).filter((node) => (node?.type || '').toLowerCase() === 'corridor').map((node) => node.id)
    );
    const targetIds = new Set(graphSelection.filter((id) => corridorIds.has(id)));
    if (targetIds.size === 0) {
      return;
    }
    setGraphState((prev) => ({
      ...prev,
      nodes: (prev.nodes ?? []).filter((node) => !targetIds.has(node.id)),
      edges: (prev.edges ?? []).filter((edge) => !targetIds.has(edge.source) && !targetIds.has(edge.target)),
    }));
    setGraphSelection((current) => current.filter((id) => !targetIds.has(id)));
    setGraphDirty(true);
  }, [graphSelection, graphState.nodes]);

  const handleGraphNodeSelect = useCallback((nodeId) => {
    setGraphSelection((prev) => {
      if (!nodeId) {
        return [];
      }
      if (prev.length === 0) {
        return [nodeId];
      }
      if (prev.length === 1) {
        if (prev[0] === nodeId) {
          return [nodeId];
        }
        return [prev[0], nodeId];
      }
      return [nodeId];
    });
  }, []);

  const handleCanvasSelectEntity = useCallback(
    (entity) => {
      if (!entity) {
        setGraphSelection([]);
        return;
      }
      if (entity.type === 'graph-node' || entity.type === 'door') {
        handleGraphNodeSelect(entity.nodeId);
      }
    },
    [handleGraphNodeSelect]
  );

  const handleCanvasSelectEdge = useCallback(({ source, target }) => {
    if (!source || !target) {
      return;
    }
    setGraphSelection([source, target]);
  }, []);

  const clearNodeSelection = () => setGraphSelection([]);

  const hasDirectEdge = useCallback(
    (a, b, edges = graphState.edges) =>
      edges.some((edge) => (edge.source === a && edge.target === b) || (edge.source === b && edge.target === a)),
    [graphState.edges]
  );

  const hasDoorCorridorBridge = useCallback(
    (doorId, corridorId, edges = graphState.edges) => {
      if (!doorId || !corridorId || !corridorId.startsWith('corridor')) {
        return false;
      }
      const endpoints = doorEndpointMap.get(doorId);
      if (!endpoints || endpoints.size === 0) {
        return false;
      }
      return edges.some(
        (edge) =>
          (edge.source === corridorId && endpoints.has(edge.target)) ||
          (edge.target === corridorId && endpoints.has(edge.source))
      );
    },
    [doorEndpointMap, graphState.edges]
  );

  const isSelectedEdgeConnected = useMemo(() => {
    if (graphSelection.length !== 2) {
      return false;
    }
    const [a, b] = graphSelection;
    if (hasDirectEdge(a, b)) {
      return true;
    }
    if (a.startsWith('door_') && b.startsWith('corridor')) {
      return hasDoorCorridorBridge(a, b);
    }
    if (b.startsWith('door_') && a.startsWith('corridor')) {
      return hasDoorCorridorBridge(b, a);
    }
    return false;
  }, [graphSelection, hasDoorCorridorBridge, hasDirectEdge]);

  const toggleEdgeBetweenSelectedNodes = () => {
    if (graphSelection.length !== 2) {
      // eslint-disable-next-line no-alert
      alert('연결을 변경할 두 개의 노드를 선택해 주세요.');
      return;
    }
    const [a, b] = graphSelection;
    const isDoorCorridorPair =
      (a.startsWith('door_') && b.startsWith('corridor')) || (b.startsWith('door_') && a.startsWith('corridor'));
    const doorId = a.startsWith('door_') ? a : b.startsWith('door_') ? b : null;
    const corridorId = a.startsWith('corridor') ? a : b.startsWith('corridor') ? b : null;

    setGraphState((prev) => {
      const edges = [...prev.edges];
      const existingIndex = edges.findIndex(
        (edge) => (edge.source === a && edge.target === b) || (edge.source === b && edge.target === a)
      );
      if (existingIndex >= 0) {
        edges.splice(existingIndex, 1);
        return { ...prev, edges };
      }

      if (isDoorCorridorPair && doorId && corridorId && hasDoorCorridorBridge(doorId, corridorId, edges)) {
        const endpoints = doorEndpointMap.get(doorId);
        if (endpoints && endpoints.size > 0) {
          const filtered = edges.filter(
            (edge) =>
              !(
                (edge.source === corridorId && endpoints.has(edge.target)) ||
                (edge.target === corridorId && endpoints.has(edge.source))
              )
          );
          return { ...prev, edges: filtered };
        }
      }

      if (isDoorCorridorPair && doorId && corridorId) {
        return {
          ...prev,
          edges: [
            ...edges,
            {
              source: corridorId,
              target: doorId,
              weight: 1,
              weightText: '1',
              attributes: {},
            },
          ],
        };
      }

      return {
        ...prev,
        edges: [
          ...edges,
          {
            source: a,
            target: b,
            weight: 1,
            weightText: '1',
            attributes: {},
          },
        ],
      };
    });
    setGraphDirty(true);
  };

  const graphSummary = useMemo(() => {
    const nodes = metadata?.graph_summary?.nodes ?? graphState.nodes.length;
    const edges = metadata?.graph_summary?.edges ?? graphState.edges.length;
    return { nodes, edges };
  }, [metadata, graphState]);

  const crossFloorEdges = useMemo(
    () =>
      (graphState.edges ?? [])
        .map((edge, index) => ({
          ...edge,
          edgeIndex: index,
        }))
        .filter(
          (edge) =>
            edge?.attributes?.is_cross_floor &&
            (edge?.attributes?.is_base_cross_floor ||
              edge?.attributes?.generated_cross_floor === undefined ||
              edge?.attributes?.generated_cross_floor === false)
        ),
    [graphState.edges]
  );

  const selectedTargetFloorOption = useMemo(
    () => floorOptions.find((option) => option.requestId === selectedTargetFloorId) ?? null,
    [floorOptions, selectedTargetFloorId]
  );

  const remoteConnectorOptions =
    remoteConnectorData.requestId === selectedTargetFloorId ? (remoteConnectorData[crossFloorType] ?? []) : [];
  const floorLabelMap = useMemo(() => {
    const merged = new Map();
    floorSummaryMap.forEach((value, key) => {
      const label = value?.floorLabel ?? value?.label ?? value?.floorValue ?? '';
      if (label) {
        merged.set(key, label);
      }
    });
    floorOptions.forEach((option) => {
      const requestKey = option?.requestId ? String(option.requestId) : null;
      if (!requestKey || merged.has(requestKey)) {
        return;
      }
      const optionLabel =
        option?.floorLabel ??
        option?.label ??
        option?.fileName ??
        `요청 ${requestKey}`;
      if (optionLabel) {
        merged.set(requestKey, optionLabel);
      }
    });
    return merged;
  }, [floorOptions, floorSummaryMap]);

  const isCreateCrossFloorDisabled =
    isRemoteConnectorLoading || !selectedLocalConnectorId || !selectedTargetFloorId || !selectedRemoteConnectorId;

  const hasCorridorSelection = useMemo(() => {
    const corridorIds = new Set(
      (graphState.nodes ?? []).filter((node) => (node?.type || '').toLowerCase() === 'corridor').map((node) => node.id)
    );
    return graphSelection.some((id) => corridorIds.has(id));
  }, [graphState.nodes, graphSelection]);

  if (!requestId) {
    return (
      <div className={styles.centerWrapper}>
        <p>요청 ID가 필요합니다.</p>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <button type='button' className={styles.backButton} onClick={handleBack}>
            <ArrowLeft size={16} /> 업로드 목록
          </button>
          <h1>그래프 편집</h1>
          <p className={styles.summary}>
            {currentFloorLabel ? `${currentFloorLabel} · ` : ''}
            요청 ID {requestId} · 노드 {graphSummary.nodes} · 엣지 {graphSummary.edges}
          </p>
        </div>
        <div className={styles.toolbar}>
          <button type='button' className={styles.secondaryButton} onClick={loadGraph} disabled={isLoading}>
            <RefreshCw size={16} /> 새로고침
          </button>
          <button type='button' className={styles.primaryButton} onClick={handleSave} disabled={isSaving}>
            <Save size={16} /> {isSaving ? '저장 중...' : '그래프 저장'}
          </button>
        </div>
      </header>

      {error && <div className={styles.errorBox}>{error}</div>}
      {graphDirty && <div className={styles.noticeBox}>저장되지 않은 그래프 변경 사항이 있습니다.</div>}

      <section className={styles.content}>
        <div className={styles.canvasColumn}>
          <StepTwoCanvas
            imageSize={imageSize}
            graph={graphPayload}
            rooms={roomEntities}
            doors={doorEntities}
            wallLines={wallLineEntities}
            stairs={stairEntities}
            elevators={elevatorEntities}
            selectedEntity={null}
            onSelectEntity={handleCanvasSelectEntity}
            isLoading={isLoading}
            error={error}
            showRoomLabels
            showDoorLabels
            enableNodeDrag
            onNodePositionChange={handleNodePositionChange}
            draggableNodeTypes={['corridor']}
            selectedGraphNodeIds={graphSelection}
            onSelectEdge={handleCanvasSelectEdge}
          />
        </div>
        <div className={styles.controlColumn}>
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <h2>간선 추가/삭제</h2>
            </div>
            <p className={styles.helperText}>
              캔버스에서 노드를 클릭하면 선택되고, 두 개를 선택하면 아래 버튼으로 연결을 추가하거나 삭제할 수 있습니다.
              드래그로 위치를 옮기고 연결을 정리해 보세요.
            </p>
            <div className={styles.selectionStatus}>
              <span>선택된 노드:</span>
              {graphSelection.length === 0 && <strong>없음</strong>}
              {graphSelection.length > 0 && <strong>{graphSelection.join(' , ')}</strong>}
            </div>
            <div className={styles.selectionButtons}>
              <button type='button' className={styles.secondaryButton} onClick={clearNodeSelection}>
                선택 초기화
              </button>
              <button
                type='button'
                className={styles.primaryButton}
                onClick={toggleEdgeBetweenSelectedNodes}
                disabled={graphSelection.length !== 2}
              >
                {isSelectedEdgeConnected ? '연결 해제' : '연결 추가'}
              </button>
            </div>
          </div>
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <h2>노드 관리</h2>
            </div>
            <p className={styles.helperText}>
              새 노드는 평면도 중앙에 생성됩니다. 삭제 기능은 선택된 모든 노드와의 연결을 함께 제거하니 주의해 주세요.
            </p>
            <div className={styles.selectionButtons}>
              <button type='button' className={styles.secondaryButton} onClick={() => handleAddNode('corridor')}>
                노드 추가
              </button>
              <button
                type='button'
                className={styles.dangerButton}
                onClick={handleDeleteSelectedNodes}
                disabled={!hasCorridorSelection}
              >
                선택 노드 삭제
              </button>
            </div>
          </div>
          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <h2>층별 연결</h2>
              <button type='button' className={styles.secondaryButton} onClick={handleOpenCrossFloorModal}>
                연결 추가
              </button>
            </div>
            {isLoadingFloorSummaries && <p className={styles.helperText}>층 정보를 불러오는 중...</p>}
            {floorSummaryError && <p className={styles.helperText}>{floorSummaryError}</p>}
            {crossFloorEdges.length === 0 && (
              <p className={styles.helperText}>
                다른 층 도면의 계단과 엘리베이터를 연결하려면 연결 추가 버튼을 눌러주세요.
              </p>
            )}
            {crossFloorEdges.length > 0 && (
              <ul className={styles.crossConnectionList}>
                {crossFloorEdges.map((edge) => {
                  const edgeKey = `${edge.source}-${edge.attributes?.target_request_id}-${edge.attributes?.target_node_id}-${edge.edgeIndex}`;
                  const targetRequestId = edge.attributes?.target_request_id;
                  const targetFloorLabel =
                    (targetRequestId && floorLabelMap.get(String(targetRequestId))) || null;
                  const targetNodeLabel = edge.attributes?.target_node_label ?? edge.attributes?.target_node_id;
                  const localLabel = currentFloorLabel ? `${currentFloorLabel} · ${edge.source}` : edge.source;
                  const remoteLabel = targetFloorLabel
                    ? `${targetFloorLabel} · ${targetNodeLabel ?? '노드'}`
                    : targetRequestId
                      ? `도면 ${targetRequestId} · ${targetNodeLabel ?? '노드'}`
                      : targetNodeLabel ?? '연결 대상 노드';
                  return (
                    <li key={edgeKey} className={styles.crossConnectionItem}>
                      <div className={styles.crossConnectionMeta}>
                        <strong>{localLabel}</strong>
                        <span>→ {remoteLabel}</span>
                      </div>
                      <button
                        type='button'
                        className={styles.dangerButton}
                        onClick={() => handleRemoveCrossFloorEdge(edge.edgeIndex)}
                      >
                        삭제
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      </section>
      {isCrossFloorModalOpen && (
        <div className={styles.modalOverlay}>
          <div className={styles.modalCard}>
            <div className={styles.modalHeader}>
              <div>
                <h3>계단·엘리베이터 연결</h3>
                <p>다른 층 도면의 동일한 유형 노드와 연결해 층간 이동 경로를 정의합니다.</p>
              </div>
              <button type='button' className={styles.iconButton} onClick={handleCloseCrossFloorModal}>
                <X size={16} />
              </button>
            </div>
            <div className={styles.modalBody}>
              <div className={styles.modalSection}>
                <span className={styles.modalLabel}>연결 유형</span>
                <div className={styles.segmentedControl}>
                  {[
                    { value: 'stair', label: '계단' },
                    { value: 'elevator', label: '엘리베이터' },
                  ].map((option) => (
                    <button
                      key={option.value}
                      type='button'
                      className={`${styles.segmentedButton} ${crossFloorType === option.value ? styles.segmentedButtonActive : ''}`}
                      onClick={() => handleChangeConnectorType(option.value)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className={styles.modalSection}>
                <span className={styles.modalLabel}>현재 도면 노드</span>
                {localConnectorNodes[crossFloorType]?.length === 0 && (
                  <p className={styles.modalHelperText}>
                    {crossFloorType === 'stair' ? '계단 노드가 없습니다.' : '엘리베이터 노드가 없습니다.'}
                  </p>
                )}
                <div className={styles.nodeList}>
                  {localConnectorNodes[crossFloorType]?.map((node) => (
                    <button
                      key={node.id}
                      type='button'
                      className={`${styles.nodeItem} ${selectedLocalConnectorId === node.id ? styles.nodeItemSelected : ''}`}
                      onClick={() => setSelectedLocalConnectorId(node.id)}
                    >
                      {node.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className={styles.modalSection}>
                <span className={styles.modalLabel}>연결할 도면</span>
                {isLoadingFloorOptions && <p className={styles.modalHelperText}>도면 목록을 불러오는 중...</p>}
                <select
                  className={styles.modalSelect}
                  value={selectedTargetFloorId ?? ''}
                  onChange={(event) => handleSelectTargetFloor(event.target.value || null)}
                >
                  <option value=''>도면을 선택하세요</option>
                  {floorOptions.map((option) => (
                    <option key={option.requestId} value={option.requestId}>
                      {option.label} (요청 {option.requestId})
                    </option>
                  ))}
                </select>
              </div>
              <div className={styles.modalSection}>
                <span className={styles.modalLabel}>
                  대상 도면 노드 {selectedTargetFloorOption ? `(요청 ${selectedTargetFloorOption.requestId})` : ''}
                </span>
                {!selectedTargetFloorId && <p className={styles.modalHelperText}>먼저 연결할 도면을 선택해주세요.</p>}
                {selectedTargetFloorId && isRemoteConnectorLoading && (
                  <p className={styles.modalHelperText}>노드 정보를 불러오는 중...</p>
                )}
                {selectedTargetFloorId && remoteConnectorError && (
                  <p className={styles.modalError}>{remoteConnectorError}</p>
                )}
                {selectedTargetFloorId &&
                  !isRemoteConnectorLoading &&
                  !remoteConnectorError &&
                  remoteConnectorOptions.length === 0 && (
                    <p className={styles.modalHelperText}>선택한 도면에 사용할 수 있는 노드가 없습니다.</p>
                  )}
                <div className={styles.nodeList}>
                  {remoteConnectorOptions.map((node) => (
                    <button
                      key={node.id}
                      type='button'
                      className={`${styles.nodeItem} ${selectedRemoteConnectorId === node.id ? styles.nodeItemSelected : ''}`}
                      onClick={() => setSelectedRemoteConnectorId(node.id)}
                    >
                      {node.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className={styles.modalFooter}>
              {crossFloorModalError && <p className={styles.modalError}>{crossFloorModalError}</p>}
              <div className={styles.modalActions}>
                <button type='button' className={styles.secondaryButton} onClick={handleCloseCrossFloorModal}>
                  취소
                </button>
                <button
                  type='button'
                  className={styles.primaryButton}
                  onClick={handleCreateCrossFloorConnection}
                  disabled={isCreateCrossFloorDisabled}
                >
                  연결 추가
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminGraphEditorPage;
