import { useEffect, useMemo, useState } from 'react';
import PropTypes from 'prop-types';
import { ArrowLeft, Loader2, Save } from 'lucide-react';
import AnnotationCanvas from '../components/annotations/AnnotationCanvas';
import AnnotationSidebar from '../components/annotations/AnnotationSidebar';
import { getDefaultLabelId, isLineLabel, isPointLabel } from '../config/annotationConfig';
import { subtractBoxFromLines } from '../utils/wallTrimmer';
import styles from './FloorPlanEditorPage.module.css';

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

const FloorPlanEditorPage = ({
  fileName,
  imageUrl,
  initialBoxes,
  initialLines,
  initialPoints,
  onCancel,
  onSubmit,
  isSaving,
  onBoxesChange,
  onLinesChange,
  onPointsChange,
  errorMessage,
}) => {
  const [boxes, setBoxes] = useState(initialBoxes);
  const [lines, setLines] = useState(initialLines);
  const [points, setPoints] = useState(initialPoints);
  const [selectedItem, setSelectedItem] = useState(() =>
    createSelectionFromData(initialBoxes, initialLines, initialPoints)
  );
  const [addMode, setAddMode] = useState(false);
  const [activeLabelId, setActiveLabelId] = useState(getDefaultLabelId());
  const [hiddenLabelIds, setHiddenLabelIds] = useState(() => new Set());

  useEffect(() => {
    setBoxes(initialBoxes);
  }, [initialBoxes]);

  useEffect(() => {
    setLines(initialLines);
  }, [initialLines]);

  useEffect(() => {
    setPoints(initialPoints);
  }, [initialPoints]);

  useEffect(() => {
    setSelectedItem((prev) => {
      if (!prev) {
        return createSelectionFromData(initialBoxes, initialLines, initialPoints);
      }

      if (prev.type === 'box' && initialBoxes.some((box) => box.id === prev.id)) {
        return prev;
      }
      if (prev.type === 'line' && initialLines.some((line) => line.id === prev.id)) {
        return prev;
      }
      if (prev.type === 'point' && initialPoints.some((point) => point.id === prev.id)) {
        return prev;
      }
      return createSelectionFromData(initialBoxes, initialLines, initialPoints);
    });
  }, [initialBoxes, initialLines, initialPoints]);

  const boxesMemo = useMemo(() => boxes, [boxes]);
  const linesMemo = useMemo(() => lines, [lines]);
  const pointsMemo = useMemo(() => points, [points]);

  useEffect(() => {
    if (!selectedItem) {
      return;
    }
    const exists =
      (selectedItem.type === 'box' && boxesMemo.some((box) => box.id === selectedItem.id)) ||
      (selectedItem.type === 'line' && linesMemo.some((line) => line.id === selectedItem.id)) ||
      (selectedItem.type === 'point' && pointsMemo.some((point) => point.id === selectedItem.id));

    if (!exists) {
      const fallback = createSelectionFromData(boxesMemo, linesMemo, pointsMemo);
      setSelectedItem(fallback);
    }
  }, [boxesMemo, linesMemo, pointsMemo, selectedItem]);

  const applyPointsUpdate = (updater) => {
    setPoints((prev) => {
      const next = updater(prev);
      onPointsChange?.(next);
      return next;
    });
  };

  const handleUpdateBoxes = (producer) => {
    setBoxes((prev) => {
      const next = producer(prev);
      onBoxesChange?.(next);
      applyPointsUpdate((prevPoints) => recalcPointsForGeometry(prevPoints, next, linesMemo));
      return next;
    });
  };

  const handleUpdateLines = (producer) => {
    setLines((prev) => {
      const next = producer(prev);
      onLinesChange?.(next);
      applyPointsUpdate((prevPoints) => recalcPointsForGeometry(prevPoints, boxesMemo, next));
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

  const handleAddShape = (draft) => {
    if (draft.type === 'line') {
      const nextId = `line-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      const prepared = { ...draft, id: nextId, labelId: draft.labelId ?? activeLabelId };
      handleUpdateLines((prev) => [...prev, prepared]);
      setSelectedItem({ type: 'line', id: nextId });
    } else if (draft.type === 'box') {
      const nextId = `box-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      const prepared = { ...draft, id: nextId, labelId: draft.labelId ?? activeLabelId };
      handleUpdateBoxes((prev) => [...prev, prepared]);
      handleUpdateLines((prev) => subtractBoxFromLines(prev, prepared));
      setSelectedItem({ type: 'box', id: nextId });
    } else if (draft.type === 'point') {
      const nextId = `point-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      const prepared = { ...draft, id: nextId, labelId: draft.labelId ?? activeLabelId };
      applyPointsUpdate((prev) => [...prev, prepared]);
      setSelectedItem({ type: 'point', id: nextId });
    }
    setAddMode(false);
  };

  const handleDelete = (item) => {
    if (!item) {
      return;
    }

    if (item.type === 'line') {
      handleUpdateLines((prev) => {
        const filtered = prev.filter((line) => line.id !== item.id);
        const remainingPoints = pointsMemo.filter(
          (point) => !(point.anchor?.type === 'line' && point.anchor?.id === item.id)
        );
        if (selectedItem?.type === 'line' && selectedItem.id === item.id) {
          const fallback = createSelectionFromData(boxesMemo, filtered, remainingPoints);
          setSelectedItem(fallback);
        }
        return filtered;
      });
    } else if (item.type === 'box') {
      const remainingBoxes = boxesMemo.filter((box) => box.id !== item.id);
      const remainingPoints = pointsMemo.filter(
        (point) => !(point.anchor?.type === 'box' && point.anchor?.id === item.id)
      );
      handleUpdateBoxes((prev) => prev.filter((box) => box.id !== item.id));
      if (selectedItem?.type === 'box' && selectedItem.id === item.id) {
        const fallback = createSelectionFromData(remainingBoxes, linesMemo, remainingPoints);
        setSelectedItem(fallback);
      }
    } else if (item.type === 'point') {
      applyPointsUpdate((prevPoints) => prevPoints.filter((point) => point.id !== item.id));
      if (selectedItem?.type === 'point' && selectedItem.id === item.id) {
        const remaining = pointsMemo.filter((point) => point.id !== item.id);
        const fallback = createSelectionFromData(boxesMemo, linesMemo, remaining);
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

  const handleSubmit = () => {
    onSubmit?.({ boxes: boxesMemo, lines: linesMemo, points: pointsMemo });
  };

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

  const selectedBox = selectedItem?.type === 'box' ? boxesMemo.find((box) => box.id === selectedItem.id) : null;
  const selectedLine = selectedItem?.type === 'line' ? linesMemo.find((line) => line.id === selectedItem.id) : null;
  const selectedPoint = selectedItem?.type === 'point' ? pointsMemo.find((point) => point.id === selectedItem.id) : null;

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
        <button type='button' className={styles.primaryButton} onClick={handleSubmit} disabled={isSaving}>
          {isSaving ? <Loader2 className={styles.spinner} size={16} /> : <Save size={16} />}
          {isSaving ? '저장 중...' : '다음 단계'}
        </button>
      </header>
      {errorMessage && <p className={styles.errorMessage}>{errorMessage}</p>}

      <main className={styles.main}>
        <section className={styles.canvasContainer}>
          <AnnotationCanvas
            imageUrl={imageUrl}
            boxes={boxesMemo}
            lines={linesMemo}
            points={pointsMemo}
            selectedItem={selectedItem}
            onSelect={handleSelect}
            onUpdateBox={handleUpdateBox}
            onUpdateLine={handleUpdateLine}
            addMode={addMode}
            activeLabelId={activeLabelId}
            onAddShape={handleAddShape}
            hiddenLabelIds={hiddenLabelIds}
          />
        </section>
        <AnnotationSidebar
          boxes={boxesMemo}
          lines={linesMemo}
          points={pointsMemo}
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
        />
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
  onCancel: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
  isSaving: PropTypes.bool,
  onBoxesChange: PropTypes.func,
  onLinesChange: PropTypes.func,
  onPointsChange: PropTypes.func,
  errorMessage: PropTypes.string,
};

FloorPlanEditorPage.defaultProps = {
  fileName: 'floor-plan.png',
  initialBoxes: [],
  initialLines: [],
  initialPoints: [],
  isSaving: false,
  onBoxesChange: undefined,
  onLinesChange: undefined,
  onPointsChange: undefined,
  errorMessage: null,
};

export default FloorPlanEditorPage;
