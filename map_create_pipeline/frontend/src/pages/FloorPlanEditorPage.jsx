import { useEffect, useRef, useState } from 'react';
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
  const canvasRef = useRef(null);

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
      applyPointsUpdate((prevPoints) => recalcPointsForGeometry(prevPoints, next, lines));
      return next;
    });
  };

  const handleUpdateLines = (producer) => {
    setLines((prev) => {
      const next = producer(prev);
      onLinesChange?.(next);
      applyPointsUpdate((prevPoints) => recalcPointsForGeometry(prevPoints, boxes, next));
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
      handleUpdateLines((prev) => subtractBoxFromLines(prev, prepared));
      setSelectedItem({ type: 'box', id: nextId });
    } else if (draft.type === 'point') {
      const nextId = `point-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      const prepared = { ...draft, id: nextId, labelId: draft.labelId ?? activeLabelId };
      applyPointsUpdate((prev) => [...prev, prepared]);
      setSelectedItem({ type: 'point', id: nextId });
    }
  };

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
          const fallback = createSelectionFromData(boxes, filtered, remainingPoints);
          setSelectedItem(fallback);
        }
        return filtered;
      });
    } else if (item.type === 'box') {
      const remainingBoxes = boxes.filter((box) => box.id !== item.id);
      const remainingPoints = points.filter(
        (point) => !(point.anchor?.type === 'box' && point.anchor?.id === item.id)
      );
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

  const handleSubmit = () => {
    onSubmit?.({ boxes, lines, points });
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
        <button type='button' className={styles.primaryButton} onClick={handleSubmit} disabled={isSaving}>
          {isSaving ? <Loader2 className={styles.spinner} size={16} /> : <Save size={16} />}
          {isSaving ? '저장 중...' : '다음 단계'}
        </button>
      </header>
      {errorMessage && <p className={styles.errorMessage}>{errorMessage}</p>}

      <main className={styles.main}>
        <section className={styles.canvasContainer}>
          <AnnotationCanvas
            ref={canvasRef}
            imageUrl={imageUrl}
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
          />
        </section>
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
