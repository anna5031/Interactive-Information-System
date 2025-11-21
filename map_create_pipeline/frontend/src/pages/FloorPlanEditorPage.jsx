import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import PropTypes from 'prop-types';
import { ArrowLeft, Eye, EyeOff, Layers, Loader2, Save, Sparkles } from 'lucide-react';
import { autoCorrectLayout, fetchFreeSpacePreview } from '../api/floorPlans';
import AnnotationCanvas from '../components/annotations/AnnotationCanvas';
import AnnotationSidebar from '../components/annotations/AnnotationSidebar';
import { getDefaultLabelId, isLineLabel, isPointLabel } from '../config/annotationConfig';
import { subtractBoxFromLines } from '../utils/wallTrimmer';
import { filterWallLinesByLength } from '../utils/wallFilter';
import styles from './FloorPlanEditorPage.module.css';

const DELETE_TRIMMED_WALLS_ON_BOX_ADD = false; // toggle-only flag for short wall deletions after clipping
const PREVIEW_OVERLAY_OPACITY = 0.45;

const toSnakeCase = (value) => value.replace(/[A-Z]/g, (match) => `_${match.toLowerCase()}`);

const decodeBase64ToUint8Array = (encoded) => {
  if (!encoded || typeof encoded !== 'string') {
    return null;
  }
  if (typeof window !== 'undefined' && typeof window.atob === 'function') {
    const binary = window.atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }
  if (typeof Buffer !== 'undefined') {
    return Uint8Array.from(Buffer.from(encoded, 'base64'));
  }
  return null;
};

const renderBitmaskToDataUrl = (maskPayload, widthHint, heightHint) => {
  if (!maskPayload || typeof maskPayload.data !== 'string' || typeof document === 'undefined') {
    return null;
  }
  const bytes = decodeBase64ToUint8Array(maskPayload.data);
  if (!bytes) {
    return null;
  }
  const shape = Array.isArray(maskPayload.shape) ? maskPayload.shape : [];
  const rawRows = Number.isFinite(shape[0]) && shape[0] > 0 ? Math.round(shape[0]) : null;
  const rawCols = Number.isFinite(shape[1]) && shape[1] > 0 ? Math.round(shape[1]) : null;
  const rows = rawRows || (Number.isFinite(heightHint) && heightHint > 0 ? Math.round(heightHint) : 1);
  const cols = rawCols || (Number.isFinite(widthHint) && widthHint > 0 ? Math.round(widthHint) : 1);
  const totalPixels = rows * cols;
  const expectedBits = Number.isFinite(maskPayload.length)
    ? Math.min(Math.max(1, Math.round(maskPayload.length)), totalPixels)
    : totalPixels;
  const canvas = document.createElement('canvas');
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return null;
  }
  const imageData = ctx.createImageData(cols, rows);
  let bitIndex = 0;
  for (let byteIndex = 0; byteIndex < bytes.length && bitIndex < expectedBits; byteIndex += 1) {
    const byte = bytes[byteIndex];
    for (let bit = 0; bit < 8 && bitIndex < expectedBits; bit += 1) {
      const value = (byte >> bit) & 1;
      const outputIndex = bitIndex * 4;
      const color = value ? 255 : 0;
      imageData.data[outputIndex] = color;
      imageData.data[outputIndex + 1] = color;
      imageData.data[outputIndex + 2] = color;
      imageData.data[outputIndex + 3] = 255;
      bitIndex += 1;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  const targetWidth = Number.isFinite(widthHint) && widthHint > 0 ? Math.round(widthHint) : cols;
  const targetHeight = Number.isFinite(heightHint) && heightHint > 0 ? Math.round(heightHint) : rows;
  if (targetWidth !== cols || targetHeight !== rows) {
    const scaled = document.createElement('canvas');
    scaled.width = targetWidth;
    scaled.height = targetHeight;
    const scaledCtx = scaled.getContext('2d');
    if (!scaledCtx) {
      return canvas.toDataURL('image/png');
    }
    scaledCtx.imageSmoothingEnabled = false;
    scaledCtx.drawImage(canvas, 0, 0, targetWidth, targetHeight);
    return scaled.toDataURL('image/png');
  }
  return canvas.toDataURL('image/png');
};

const resolvePreviewOverlayUrl = (preview, maskKey, fallbackWidth, fallbackHeight) => {
  if (!preview) {
    return null;
  }
  const directValue = preview?.[maskKey];
  if (typeof directValue === 'string') {
    return directValue;
  }
  const snakeValue = preview?.[toSnakeCase(maskKey)];
  if (typeof snakeValue === 'string') {
    return snakeValue;
  }
  const artifactBundle = preview?.artifactBundle ?? preview?.artifact_bundle;
  if (!artifactBundle?.masks) {
    return null;
  }
  const maskMap = {
    freeSpaceMask: 'freeSpace',
    doorMask: 'door',
    roomMask: 'room',
    wallMask: 'wall',
  };
  const artifactKey = maskMap[maskKey];
  if (!artifactKey) {
    return null;
  }
  const maskPayload = artifactBundle.masks[artifactKey];
  if (!maskPayload) {
    return null;
  }
  const sizePayload = preview.imageSize || preview.image_size || {};
  const resolvedWidth =
    (Number.isFinite(artifactBundle.width) && artifactBundle.width > 0 && Math.round(artifactBundle.width)) ||
    (Number.isFinite(sizePayload.width) && sizePayload.width > 0 && Math.round(sizePayload.width)) ||
    (Number.isFinite(fallbackWidth) && fallbackWidth > 0 && Math.round(fallbackWidth)) ||
    (Array.isArray(maskPayload.shape) &&
      Number.isFinite(maskPayload.shape[1]) &&
      maskPayload.shape[1] > 0 &&
      Math.round(maskPayload.shape[1])) ||
    null;
  const resolvedHeight =
    (Number.isFinite(artifactBundle.height) && artifactBundle.height > 0 && Math.round(artifactBundle.height)) ||
    (Number.isFinite(sizePayload.height) && sizePayload.height > 0 && Math.round(sizePayload.height)) ||
    (Number.isFinite(fallbackHeight) && fallbackHeight > 0 && Math.round(fallbackHeight)) ||
    (Array.isArray(maskPayload.shape) &&
      Number.isFinite(maskPayload.shape[0]) &&
      maskPayload.shape[0] > 0 &&
      Math.round(maskPayload.shape[0])) ||
    null;
  const width = resolvedWidth || 1;
  const height = resolvedHeight || 1;
  return renderBitmaskToDataUrl(maskPayload, width, height);
};

const createSelectionFromData = (boxes, lines, points) => {
  if (boxes.length > 0) {
    return { type: 'box', id: boxes[0].id };
  }
  if (lines.length > 0) {
    return { type: 'line', id: lines[0].id };
  }
  if (points.length > 0) {
    return { type: 'point', id: points[0].id };
  }
  return null;
};

const recalcPointsForGeometry = (points, boxes, lines) => {
  if (!Array.isArray(points) || points.length === 0) {
    return points;
  }

  const boxesMap = new Map(boxes.map((box) => [box.id, box]));
  const linesMap = new Map(lines.map((line) => [line.id, line]));

  return points
    .map((point) => {
      const { anchor } = point || {};
      if (!anchor) {
        return point;
      }

      if (anchor.type === 'line') {
        const line = linesMap.get(anchor.id);
        if (!line) {
          return null;
        }
        const t = Number.isFinite(anchor.t) ? Math.max(0, Math.min(1, anchor.t)) : 0;
        const x = line.x1 + (line.x2 - line.x1) * t;
        const y = line.y1 + (line.y2 - line.y1) * t;
        const lineIndex = lines.indexOf(line);
        return {
          ...point,
          x,
          y,
          anchor: {
            type: 'line',
            id: line.id,
            index: lineIndex >= 0 ? lineIndex : anchor.index,
            t,
          },
        };
      }

      if (anchor.type === 'box') {
        const box = boxesMap.get(anchor.id);
        if (!box) {
          return null;
        }
        const t = Number.isFinite(anchor.t) ? Math.max(0, Math.min(1, anchor.t)) : 0;
        let x = point.x;
        let y = point.y;

        switch (anchor.edge) {
          case 'top':
            y = box.y;
            x = box.x + box.width * t;
            break;
          case 'bottom':
            y = box.y + box.height;
            x = box.x + box.width * t;
            break;
          case 'left':
            x = box.x;
            y = box.y + box.height * t;
            break;
          case 'right':
            x = box.x + box.width;
            y = box.y + box.height * t;
            break;
          default:
            return point;
        }

        const boxIndex = boxes.indexOf(box);
        return {
          ...point,
          x,
          y,
          anchor: {
            type: 'box',
            id: box.id,
            index: boxIndex >= 0 ? boxIndex : anchor.index,
            edge: anchor.edge,
            t,
          },
        };
      }

      return point;
    })
    .filter(Boolean);
};

const getNextWallId = (existingLines) => {
  const regex = /^wall-(\d+)$/;
  let maxIndex = -1;

  existingLines.forEach((line) => {
    const match = regex.exec(line.id);
    if (match) {
      const value = Number.parseInt(match[1], 10);
      if (!Number.isNaN(value) && value > maxIndex) {
        maxIndex = value;
      }
    }
  });

  return `wall-${maxIndex + 1}`;
};

const anchorsEqual = (left, right) => {
  if (left === right) {
    return true;
  }
  if (!left || !right) {
    return false;
  }
  if (left.type !== right.type) {
    return false;
  }
  if (left.type === 'line') {
    return left.id === right.id && left.t === right.t;
  }
  if (left.type === 'box') {
    return left.id === right.id && left.edge === right.edge && left.t === right.t;
  }
  return false;
};

const linesEqual = (a, b) => {
  if (a === b) {
    return true;
  }
  if (!a || !b) {
    return false;
  }
  return a.id === b.id && a.labelId === b.labelId && a.x1 === b.x1 && a.y1 === b.y1 && a.x2 === b.x2 && a.y2 === b.y2;
};

const boxesEqual = (a, b) => {
  if (a === b) {
    return true;
  }
  if (!a || !b) {
    return false;
  }
  return (
    a.id === b.id &&
    a.labelId === b.labelId &&
    a.x === b.x &&
    a.y === b.y &&
    a.width === b.width &&
    a.height === b.height
  );
};

const pointsEqual = (a, b) => {
  if (a === b) {
    return true;
  }
  if (!a || !b) {
    return false;
  }
  return a.id === b.id && a.labelId === b.labelId && a.x === b.x && a.y === b.y && anchorsEqual(a.anchor, b.anchor);
};

const shallowArrayEqual = (lhs, rhs, comparator) => {
  if (lhs === rhs) {
    return true;
  }
  if (!Array.isArray(lhs) || !Array.isArray(rhs)) {
    return false;
  }
  if (lhs.length !== rhs.length) {
    return false;
  }
  for (let index = 0; index < lhs.length; index += 1) {
    if (!comparator(lhs[index], rhs[index])) {
      return false;
    }
  }
  return true;
};

const ensureArray = (value) => (Array.isArray(value) ? value : []);

const useSyncedArrayState = (externalValue, comparator, onChange) => {
  const normalized = ensureArray(externalValue);
  const [internal, setInternal] = useState(normalized);
  const syncingFromExternalRef = useRef(false);
  const lastExternalValueRef = useRef(normalized);
  const lastNotifiedValueRef = useRef(normalized);

  useEffect(() => {
    const next = ensureArray(externalValue);
    if (shallowArrayEqual(lastExternalValueRef.current ?? [], next ?? [], comparator)) {
      return;
    }
    syncingFromExternalRef.current = true;
    lastExternalValueRef.current = next;
    setInternal(next);
  }, [externalValue, comparator]);

  useEffect(() => {
    if (!onChange) {
      return;
    }
    if (syncingFromExternalRef.current) {
      syncingFromExternalRef.current = false;
      lastNotifiedValueRef.current = internal;
      return;
    }
    if (shallowArrayEqual(lastNotifiedValueRef.current ?? [], internal ?? [], comparator)) {
      return;
    }
    lastNotifiedValueRef.current = internal;
    lastExternalValueRef.current = internal;
    onChange(internal);
  }, [internal, onChange, comparator]);

  return [internal, setInternal];
};

const FloorPlanEditorPage = ({
  fileName,
  imageUrl,
  initialBoxes,
  initialLines,
  initialPoints,
  imageWidth,
  imageHeight,
  initialWallBaseLines,
  onBaseLinesChange,
  wallFilter,
  onWallFilterChange,
  onCancel,
  onSubmit,
  isSaving,
  onBoxesChange,
  onLinesChange,
  onPointsChange,
  errorMessage,
  calibrationLine,
  calibrationLengthMeters,
  metersPerPixel,
  onCalibrationLineChange,
  onCalibrationLengthChange,
}) => {
  const resolvedInitialBaseLines = useMemo(
    () =>
      Array.isArray(initialWallBaseLines) && initialWallBaseLines.length > 0 ? initialWallBaseLines : initialLines,
    [initialWallBaseLines, initialLines]
  );
  const [boxes, setBoxes] = useSyncedArrayState(initialBoxes, boxesEqual, onBoxesChange);
  const [baseLines, setBaseLines] = useSyncedArrayState(resolvedInitialBaseLines, linesEqual, onBaseLinesChange);
  const [points, setPoints] = useSyncedArrayState(initialPoints, pointsEqual, onPointsChange);
  const filterResult = useMemo(() => {
    const protectedLineIds = new Set();
    (points || []).forEach((point) => {
      if (point?.anchor?.type === 'line' && point.anchor.id) {
        protectedLineIds.add(point.anchor.id);
      }
    });
    return filterWallLinesByLength(baseLines, wallFilter, imageWidth, imageHeight, {
      protectedLineIds,
    });
  }, [baseLines, wallFilter, imageWidth, imageHeight, points]);
  const lines = filterResult.lines;
  const wallFilterStats = filterResult.stats;
  const [selectedItem, setSelectedItem] = useState(() => createSelectionFromData(boxes, lines, points));
  const [addMode, setAddMode] = useState(false);
  const [activeLabelId, setActiveLabelId] = useState(getDefaultLabelId());
  const [hiddenLabelIds, setHiddenLabelIds] = useState(() => new Set());
  const [isAutoCorrecting, setIsAutoCorrecting] = useState(false);
  const [autoCorrectError, setAutoCorrectError] = useState(null);
  const [autoCorrectStats, setAutoCorrectStats] = useState(null);
  const [freeSpacePreview, setFreeSpacePreview] = useState(null);
  const [isPreviewVisible, setIsPreviewVisible] = useState(false);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [isPreviewDirty, setIsPreviewDirty] = useState(false);
  const canvasRef = useRef(null);
  const lastSyncedLinesRef = useRef(ensureArray(initialLines));
  const calibrationLengthInput = calibrationLengthMeters ?? '';
  const calibrationReady = Number.isFinite(metersPerPixel) && metersPerPixel > 0;
  const calibrationPixelLength = useMemo(() => {
    if (!calibrationLine || !Number.isFinite(imageWidth) || !Number.isFinite(imageHeight)) {
      return null;
    }
    const dx = (calibrationLine.x2 - calibrationLine.x1) * imageWidth;
    const dy = (calibrationLine.y2 - calibrationLine.y1) * imageHeight;
    const distance = Math.hypot(dx, dy);
    return Number.isFinite(distance) ? distance : null;
  }, [calibrationLine, imageWidth, imageHeight]);
  const handleCalibrationLengthInputChange = (event) => {
    onCalibrationLengthChange?.(event.target.value);
  };
  const disableSubmit = isSaving || !calibrationReady;
  const previewAvailableRef = useRef(false);
  const initialPreviewRequestedRef = useRef(false);
  const [autoCorrectToast, setAutoCorrectToast] = useState(null);
  const autoCorrectToastTimerRef = useRef(null);

  const previewCoverageText = useMemo(() => {
    if (!freeSpacePreview || typeof freeSpacePreview.freeSpaceRatio !== 'number') {
      return null;
    }
    const ratio = Math.max(0, Math.min(1, freeSpacePreview.freeSpaceRatio));
    return `${Math.round(ratio * 1000) / 10}%`;
  }, [freeSpacePreview]);

  const previewOverlayUrl = useMemo(
    () => resolvePreviewOverlayUrl(freeSpacePreview, 'freeSpaceMask', imageWidth, imageHeight),
    [freeSpacePreview, imageWidth, imageHeight]
  );

  const previewStatusLabel = useMemo(() => {
    if (!freeSpacePreview) {
      return null;
    }
    const segments = [];
    if (previewCoverageText) {
      segments.push(`자유 공간 ${previewCoverageText}`);
    }
    segments.push(isPreviewVisible ? '표시 중' : '숨김');
    if (isPreviewDirty) {
      segments.push('편집됨 - 다시 생성 필요');
    }
    return segments.join(' · ');
  }, [freeSpacePreview, previewCoverageText, isPreviewVisible, isPreviewDirty]);

  useEffect(() => {
    lastSyncedLinesRef.current = ensureArray(initialLines);
  }, [initialLines]);

  useEffect(() => {
    const previouslyAvailable = previewAvailableRef.current;
    previewAvailableRef.current = !!freeSpacePreview;
    if (!previouslyAvailable && freeSpacePreview) {
      setIsPreviewVisible(true);
    }
  }, [freeSpacePreview]);

  useEffect(() => {
    if (!previewAvailableRef.current) {
      return;
    }
    setIsPreviewDirty(true);
  }, [boxes, baseLines, points]);

  useEffect(() => {
    setSelectedItem((prev) => {
      if (!prev) {
        return createSelectionFromData(boxes, lines, points);
      }

      if (prev.type === 'box' && boxes.some((box) => box.id === prev.id)) {
        return prev;
      }
      if (prev.type === 'line' && lines.some((line) => line.id === prev.id)) {
        return prev;
      }
      if (prev.type === 'point' && points.some((point) => point.id === prev.id)) {
        return prev;
      }
      return createSelectionFromData(boxes, lines, points);
    });
  }, [boxes, lines, points]);

  useEffect(() => {
    if (!selectedItem) {
      return;
    }
    const exists =
      (selectedItem.type === 'box' && boxes.some((box) => box.id === selectedItem.id)) ||
      (selectedItem.type === 'line' && lines.some((line) => line.id === selectedItem.id)) ||
      (selectedItem.type === 'point' && points.some((point) => point.id === selectedItem.id));

    if (!exists) {
      const fallback = createSelectionFromData(boxes, lines, points);
      setSelectedItem(fallback);
    }
  }, [boxes, lines, points, selectedItem]);

  useEffect(() => {
    if (!onLinesChange) {
      return;
    }
    if (shallowArrayEqual(lastSyncedLinesRef.current ?? [], lines ?? [], linesEqual)) {
      return;
    }
    lastSyncedLinesRef.current = lines;
    onLinesChange(lines);
  }, [lines, onLinesChange]);

  const applyPointsUpdate = useCallback(
    (updater) => {
      setPoints((prev) => {
        const rawNext = updater(prev);
        const next = Array.isArray(rawNext) ? rawNext : [];
        if (shallowArrayEqual(prev, next, pointsEqual)) {
          return prev;
        }
        return next;
      });
    },
    [setPoints]
  );

  useEffect(() => {
    applyPointsUpdate((prevPoints) => recalcPointsForGeometry(prevPoints, boxes, baseLines));
  }, [applyPointsUpdate, boxes, baseLines]);

  const handleUpdateBoxes = (producer) => {
    setBoxes((prev) => {
      const next = producer(prev);
      return next;
    });
  };

  const handleUpdateLines = (producer) => {
    setBaseLines((prev) => {
      const next = producer(prev);
      return next;
    });
  };

  const handleSelect = (item) => {
    if (addMode) {
      return;
    }
    setSelectedItem(item);
  };

  const handleUpdateBox = (id, updates) => {
    handleUpdateBoxes((prev) => prev.map((box) => (box.id === id ? { ...box, ...updates } : box)));
  };

  const handleUpdateLine = (id, updates) => {
    handleUpdateLines((prev) => prev.map((line) => (line.id === id ? { ...line, ...updates } : line)));
  };

  const handleUpdatePoint = (id, updates) => {
    applyPointsUpdate((prev) => prev.map((point) => (point.id === id ? { ...point, ...updates } : point)));
  };

  const handleAddShape = (draft) => {
    if (draft.type === 'line') {
      let createdId;
      handleUpdateLines((prev) => {
        const nextId = getNextWallId(prev);
        createdId = nextId;
        const prepared = { ...draft, id: nextId, labelId: draft.labelId ?? activeLabelId };
        return [...prev, prepared];
      });
      if (createdId) {
        setSelectedItem({ type: 'line', id: createdId });
      }
    } else if (draft.type === 'box') {
      const resolvedLabelId = draft.labelId ?? activeLabelId;
      const nextIndex = (() => {
        const regex = new RegExp(`^${resolvedLabelId}-box-(\\d+)$`);
        let max = -1;
        boxes.forEach((box) => {
          const match = regex.exec(box.id);
          if (match) {
            const value = Number.parseInt(match[1], 10);
            if (!Number.isNaN(value) && value > max) {
              max = value;
            }
          }
        });
        return max + 1;
      })();
      const nextId = `${resolvedLabelId}-box-${nextIndex}`;
      const prepared = { ...draft, id: nextId, labelId: resolvedLabelId };
      handleUpdateBoxes((prev) => [...prev, prepared]);
      handleUpdateLines((prev) =>
        subtractBoxFromLines(prev, prepared, {
          cutMargin: 0,
          edgeProximityMargin: 0,
          deleteTinySegments: DELETE_TRIMMED_WALLS_ON_BOX_ADD,
        })
      );
      setSelectedItem({ type: 'box', id: nextId });
    } else if (draft.type === 'point') {
      const nextId = `point-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      const prepared = { ...draft, id: nextId, labelId: draft.labelId ?? activeLabelId };
      applyPointsUpdate((prev) => [...prev, prepared]);
      setSelectedItem({ type: 'point', id: nextId });
    }
  };

  const handleGeneratePreview = useCallback(async () => {
    if (!Number.isFinite(imageWidth) || !Number.isFinite(imageHeight) || imageWidth <= 0 || imageHeight <= 0) {
      setPreviewError('이미지 크기 정보를 확인해 주세요.');
      return null;
    }
    setIsPreviewLoading(true);
    setPreviewError(null);
    try {
      const preview = await fetchFreeSpacePreview({
        boxes,
        lines,
        baseLines,
        points,
        imageWidth,
        imageHeight,
      });
      setFreeSpacePreview(preview);
      setIsPreviewVisible(true);
      setIsPreviewDirty(false);
      return preview;
    } catch (error) {
      console.error('Failed to load free-space preview', error);
      setPreviewError(error?.message || '복도 미리보기를 불러오지 못했습니다.');
      return null;
    } finally {
      setIsPreviewLoading(false);
    }
  }, [boxes, lines, baseLines, points, imageWidth, imageHeight]);

  useEffect(() => {
    if (freeSpacePreview || isPreviewLoading || initialPreviewRequestedRef.current) {
      return;
    }
    if (!Number.isFinite(imageWidth) || !Number.isFinite(imageHeight) || imageWidth <= 0 || imageHeight <= 0) {
      return;
    }
    initialPreviewRequestedRef.current = true;
    let cancelled = false;
    (async () => {
      const preview = await handleGeneratePreview();
      if (!preview && !cancelled) {
        initialPreviewRequestedRef.current = false;
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [freeSpacePreview, isPreviewLoading, imageWidth, imageHeight, handleGeneratePreview]);

  const handleTogglePreviewVisibility = useCallback(() => {
    if (!freeSpacePreview || !previewOverlayUrl) {
      return;
    }
    setIsPreviewVisible((prev) => !prev);
  }, [freeSpacePreview, previewOverlayUrl]);

  const handleDelete = (item) => {
    if (!item) {
      return;
    }

    if (item.type === 'line') {
      handleUpdateLines((prev) => {
        const filtered = prev.filter((line) => line.id !== item.id);
        const remainingPoints = points.filter(
          (point) => !(point.anchor?.type === 'line' && point.anchor?.id === item.id)
        );
        if (selectedItem?.type === 'line' && selectedItem.id === item.id) {
          const protectedLineIds = new Set(
            (remainingPoints || [])
              .filter((point) => point?.anchor?.type === 'line' && point.anchor.id)
              .map((point) => point.anchor.id)
          );
          const filteredVisible = filterWallLinesByLength(filtered, wallFilter, imageWidth, imageHeight, {
            protectedLineIds,
          }).lines;
          const fallback = createSelectionFromData(boxes, filteredVisible, remainingPoints);
          setSelectedItem(fallback);
        }
        return filtered;
      });
    } else if (item.type === 'box') {
      const remainingBoxes = boxes.filter((box) => box.id !== item.id);
      const remainingPoints = points.filter((point) => !(point.anchor?.type === 'box' && point.anchor?.id === item.id));
      handleUpdateBoxes((prev) => prev.filter((box) => box.id !== item.id));
      if (selectedItem?.type === 'box' && selectedItem.id === item.id) {
        const fallback = createSelectionFromData(remainingBoxes, lines, remainingPoints);
        setSelectedItem(fallback);
      }
    } else if (item.type === 'point') {
      applyPointsUpdate((prevPoints) => prevPoints.filter((point) => point.id !== item.id));
      if (selectedItem?.type === 'point' && selectedItem.id === item.id) {
        const remaining = points.filter((point) => point.id !== item.id);
        const fallback = createSelectionFromData(boxes, lines, remaining);
        setSelectedItem(fallback);
      }
    }
  };

  const handleLabelChange = (item, labelId) => {
    if (!item) {
      return;
    }

    if (item.type === 'line') {
      handleUpdateLines((prev) => prev.map((line) => (line.id === item.id ? { ...line, labelId } : line)));
    } else if (item.type === 'box') {
      handleUpdateBoxes((prev) => prev.map((box) => (box.id === item.id ? { ...box, labelId } : box)));
    } else if (item.type === 'point') {
      applyPointsUpdate((prev) => prev.map((point) => (point.id === item.id ? { ...point, labelId } : point)));
    }
  };

  const handleSubmit = async () => {
    if (isSaving) {
      return;
    }
    let preview = freeSpacePreview;
    if (!preview || isPreviewDirty) {
      const generated = await handleGeneratePreview();
      if (!generated) {
        return;
      }
      preview = generated;
    }
    await onSubmit?.({
      boxes,
      lines,
      points,
      baseLines,
      wallFilterState: wallFilter,
      freeSpacePreview: preview,
    });
  };

  const handleAutoCorrect = async () => {
    if (isAutoCorrecting) {
      return;
    }
    setAutoCorrectError(null);
    setAutoCorrectStats(null);
    setIsAutoCorrecting(true);
    try {
      const result = await autoCorrectLayout({
        boxes,
        lines: baseLines,
        imageWidth,
        imageHeight,
      });
      if (Array.isArray(result?.boxes)) {
        handleUpdateBoxes(() => result.boxes);
      }
      if (Array.isArray(result?.lines)) {
        handleUpdateLines(() => result.lines);
      }
      setAutoCorrectStats(result?.stats ?? null);
    } catch (error) {
      console.error('자동 보정 요청 실패', error);
      setAutoCorrectError('자동 보정 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setIsAutoCorrecting(false);
    }
  };

  const autoCorrectMessage = useMemo(() => {
    if (!autoCorrectStats) {
      return null;
    }
    const parts = [];
    const boxFixes = autoCorrectStats.boxBoxAdjustments ?? 0;
    const boxSnaps = autoCorrectStats.wallBoxSnaps ?? 0;
    const wallSnaps = autoCorrectStats.wallWallSnaps ?? 0;
    if (boxFixes > 0) {
      parts.push(`박스 정렬 ${boxFixes}건`);
    }
    if (boxSnaps > 0) {
      parts.push(`벽-박스 스냅 ${boxSnaps}건`);
    }
    if (wallSnaps > 0) {
      parts.push(`벽-벽 스냅 ${wallSnaps}건`);
    }
    if (parts.length === 0) {
      return '적용할 자동 보정 항목을 찾지 못했습니다.';
    }
    return `자동 보정을 적용했습니다: ${parts.join(', ')}`;
  }, [autoCorrectStats]);

  useEffect(() => {
    if (!autoCorrectMessage) {
      return;
    }
    setAutoCorrectToast(autoCorrectMessage);
    if (autoCorrectToastTimerRef.current) {
      clearTimeout(autoCorrectToastTimerRef.current);
    }
    autoCorrectToastTimerRef.current = setTimeout(() => {
      setAutoCorrectToast(null);
      autoCorrectToastTimerRef.current = null;
    }, 4000);
    return () => {
      if (autoCorrectToastTimerRef.current) {
        clearTimeout(autoCorrectToastTimerRef.current);
        autoCorrectToastTimerRef.current = null;
      }
    };
  }, [autoCorrectMessage]);

  const handleToggleAddMode = () => {
    setAddMode((prev) => {
      const next = !prev;
      if (next) {
        setSelectedItem(null);
      }
      return next;
    });
  };

  const handleToggleLabelVisibility = (labelId) => {
    setHiddenLabelIds((prev) => {
      const next = new Set(prev);
      if (next.has(labelId)) {
        next.delete(labelId);
      } else {
        next.add(labelId);
      }
      return next;
    });
  };

  const selectedBox = selectedItem?.type === 'box' ? boxes.find((box) => box.id === selectedItem.id) : null;
  const selectedLine = selectedItem?.type === 'line' ? lines.find((line) => line.id === selectedItem.id) : null;
  const selectedPoint = selectedItem?.type === 'point' ? points.find((point) => point.id === selectedItem.id) : null;

  useEffect(() => {
    const handleKeyDown = (event) => {
      const target = event.target;
      if (target) {
        const tagName = target.tagName;
        const isEditable =
          tagName === 'INPUT' ||
          tagName === 'TEXTAREA' ||
          target.isContentEditable ||
          target.getAttribute?.('contenteditable') === 'true';
        if (isEditable) {
          return;
        }
      }

      if (event.key === 'Escape') {
        if (addMode) {
          event.preventDefault();
          setAddMode(false);
        }
        return;
      }

      if ((event.key === 'Backspace' || event.key === 'Delete') && selectedItem) {
        event.preventDefault();
        handleDelete(selectedItem);
        return;
      }

      if (!event.metaKey && !event.ctrlKey && !event.altKey) {
        if (event.key?.toLowerCase() === 'a') {
          event.preventDefault();
          handleToggleAddMode();
          return;
        }
      }

      if (event.metaKey || event.ctrlKey) {
        if (event.key === '=' || event.key === '+') {
          event.preventDefault();
          canvasRef.current?.zoomIn();
          return;
        }
        if (event.key === '-') {
          event.preventDefault();
          canvasRef.current?.zoomOut();
          return;
        }
        if (event.key === '0') {
          event.preventDefault();
          canvasRef.current?.resetZoom();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown, true);
    return () => {
      window.removeEventListener('keydown', handleKeyDown, true);
    };
  }, [addMode, handleDelete, handleToggleAddMode, selectedItem]);

  return (
    <div className={styles.wrapper}>
      <header className={styles.header}>
        <button type='button' className={styles.secondaryButton} onClick={onCancel}>
          <ArrowLeft size={18} />
          돌아가기
        </button>
        <div className={styles.titleGroup}>
          <h2 className={styles.title}>도면 라벨링</h2>
          <span className={styles.fileName}>{fileName}</span>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.actionGroup}>
            <button
              type='button'
              className={styles.secondaryButton}
              onClick={handleGeneratePreview}
              disabled={isPreviewLoading}
            >
              {isPreviewLoading ? <Loader2 className={styles.spinner} size={16} /> : <Layers size={16} />}
              {isPreviewLoading ? '미리보기 생성 중...' : freeSpacePreview ? '미리보기 갱신' : '복도 미리보기'}
            </button>
            {freeSpacePreview && (
              <button type='button' className={styles.secondaryButton} onClick={handleTogglePreviewVisibility}>
                {isPreviewVisible ? <EyeOff size={16} /> : <Eye size={16} />}
                {isPreviewVisible ? '미리보기 숨기기' : '미리보기 표시'}
              </button>
            )}
            <button type='button' className={styles.autoButton} onClick={handleAutoCorrect} disabled={isAutoCorrecting}>
              {isAutoCorrecting ? <Loader2 className={styles.spinner} size={16} /> : <Sparkles size={16} />}
              {isAutoCorrecting ? '보정 중...' : '자동 보정'}
            </button>
            <button type='button' className={styles.primaryButton} onClick={handleSubmit} disabled={disableSubmit}>
              {isSaving ? <Loader2 className={styles.spinner} size={16} /> : <Save size={16} />}
              {isSaving ? '저장 중...' : '다음 단계'}
            </button>
          </div>
          <div className={styles.scaleWarningSlot}>
            {calibrationReady ? null : (
              <span className={styles.scaleWarningBanner}>축척을 설정해야 다음 단계로 진행할 수 있습니다.</span>
            )}
          </div>
        </div>
      </header>
      {errorMessage && <p className={styles.errorMessage}>{errorMessage}</p>}
      {autoCorrectError && <p className={styles.errorMessage}>{autoCorrectError}</p>}
      {previewError && <p className={styles.errorMessage}>{previewError}</p>}
      {previewStatusLabel && (
        <p className={`${styles.previewMessage} ${isPreviewDirty ? styles.previewMessageWarning : ''}`}>
          복도 미리보기 · {previewStatusLabel}
        </p>
      )}

      <main className={styles.main}>
        <section className={styles.canvasContainer}>
          {autoCorrectToast && (
            <div className={styles.autoCorrectToast} role='status'>
              {autoCorrectToast}
            </div>
          )}
          <AnnotationCanvas
            ref={canvasRef}
            imageUrl={imageUrl}
            previewOverlayUrl={previewOverlayUrl}
            previewOverlayVisible={isPreviewVisible && !!previewOverlayUrl}
            previewOverlayOpacity={PREVIEW_OVERLAY_OPACITY}
            boxes={boxes}
            lines={lines}
            points={points}
            selectedItem={selectedItem}
            onSelect={handleSelect}
            onUpdateBox={handleUpdateBox}
            onUpdateLine={handleUpdateLine}
            onUpdatePoint={handleUpdatePoint}
            addMode={addMode}
            activeLabelId={activeLabelId}
            onAddShape={handleAddShape}
            hiddenLabelIds={hiddenLabelIds}
            calibrationLine={calibrationLine}
            onCalibrationLineChange={onCalibrationLineChange}
            calibrationReadOnly={isSaving}
          />
        </section>
        <aside className={styles.sidebar}>
          <section className={styles.calibrationPanel}>
            <div className={styles.calibrationDescription}>
              <div className={styles.calibrationTitle}>
                <h3>축척 설정</h3>
                <button type='button' className={styles.helpIcon} aria-label='축척 설정 안내'>
                  ?
                  <span className={styles.helpTooltip}>
                    초록색 기준선을 기준 벽 위에 위치시키고 길이를 맞춘 뒤 실제 길이를 입력하면 축척이 계산됩니다.
                    기준선은 벽과 동일하게 이동·정렬·길이 조절이 가능합니다.
                  </span>
                </button>
              </div>
            </div>
            <div className={styles.calibrationFields}>
              <label className={styles.calibrationField} htmlFor='calibration-length'>
                <span className={styles.fieldLabel}>실제 길이 (m)</span>
                <input
                  id='calibration-length'
                  type='number'
                  step='0.01'
                  min='0'
                  placeholder='예: 3.5'
                  value={calibrationLengthInput}
                  onChange={handleCalibrationLengthInputChange}
                  disabled={isSaving}
                />
              </label>
              <div className={styles.calibrationField}>
                <span className={styles.fieldLabel}>기준선 길이 (px)</span>
                <div className={styles.calibrationValueBox}>
                  <span className={styles.calibrationValue}>
                    {calibrationPixelLength != null ? calibrationPixelLength.toFixed(1) : '-'}
                  </span>
                </div>
              </div>
            </div>
            <div className={styles.calibrationScaleSummary}>
              계산된 축척: {calibrationReady ? `1px ≈ ${metersPerPixel.toFixed(4)}m` : '-'}
            </div>
          </section>
          <div className={styles.sidebarContent}>
            <AnnotationSidebar
              boxes={boxes}
              lines={lines}
              points={points}
              selectedBox={selectedBox}
              selectedLine={selectedLine}
              selectedPoint={selectedPoint}
              onSelect={handleSelect}
              onDelete={handleDelete}
              onLabelChange={handleLabelChange}
              activeLabelId={activeLabelId}
              onActiveLabelChange={setActiveLabelId}
              addMode={addMode}
              onToggleAddMode={handleToggleAddMode}
              hiddenLabelIds={hiddenLabelIds}
              onToggleLabelVisibility={handleToggleLabelVisibility}
              isLineLabelActive={isLineLabel(activeLabelId)}
              isPointLabelActive={isPointLabel(activeLabelId)}
              wallFilter={wallFilter}
              wallFilterStats={wallFilterStats}
              onWallFilterChange={onWallFilterChange}
            />
          </div>
        </aside>
      </main>
    </div>
  );
};

FloorPlanEditorPage.propTypes = {
  fileName: PropTypes.string,
  imageUrl: PropTypes.string.isRequired,
  initialBoxes: PropTypes.arrayOf(PropTypes.object),
  initialLines: PropTypes.arrayOf(PropTypes.object),
  initialPoints: PropTypes.arrayOf(PropTypes.object),
  initialWallBaseLines: PropTypes.arrayOf(PropTypes.object),
  imageWidth: PropTypes.number,
  imageHeight: PropTypes.number,
  onCancel: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
  isSaving: PropTypes.bool,
  onBoxesChange: PropTypes.func,
  onLinesChange: PropTypes.func,
  onPointsChange: PropTypes.func,
  onBaseLinesChange: PropTypes.func,
  errorMessage: PropTypes.string,
  wallFilter: PropTypes.shape({
    enabled: PropTypes.bool,
    percentile: PropTypes.number,
    minPixels: PropTypes.number,
    minSegments: PropTypes.number,
  }),
  onWallFilterChange: PropTypes.func,
  calibrationLine: PropTypes.shape({
    x1: PropTypes.number,
    y1: PropTypes.number,
    x2: PropTypes.number,
    y2: PropTypes.number,
  }),
  calibrationLengthMeters: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  metersPerPixel: PropTypes.number,
  onCalibrationLineChange: PropTypes.func,
  onCalibrationLengthChange: PropTypes.func,
};

FloorPlanEditorPage.defaultProps = {
  fileName: 'floor-plan.png',
  initialBoxes: [],
  initialLines: [],
  initialPoints: [],
  initialWallBaseLines: null,
  imageWidth: 0,
  imageHeight: 0,
  isSaving: false,
  onBoxesChange: undefined,
  onLinesChange: undefined,
  onPointsChange: undefined,
  onBaseLinesChange: undefined,
  errorMessage: null,
  wallFilter: null,
  onWallFilterChange: undefined,
  calibrationLine: null,
  calibrationLengthMeters: '',
  metersPerPixel: null,
  onCalibrationLineChange: undefined,
  onCalibrationLengthChange: undefined,
};

export default FloorPlanEditorPage;
