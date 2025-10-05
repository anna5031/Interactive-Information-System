import { useEffect, useMemo, useState } from 'react';
import PropTypes from 'prop-types';
import { ArrowLeft, Loader2, Save } from 'lucide-react';
import AnnotationCanvas from '../components/annotations/AnnotationCanvas';
import AnnotationSidebar from '../components/annotations/AnnotationSidebar';
import { getDefaultLabelId, isLineLabel } from '../config/annotationConfig';
import { subtractBoxFromLines } from '../utils/wallTrimmer';
import styles from './FloorPlanEditorPage.module.css';

const createSelectionFromData = (boxes, lines) => {
  if (boxes.length > 0) {
    return { type: 'box', id: boxes[0].id };
  }
  if (lines.length > 0) {
    return { type: 'line', id: lines[0].id };
  }
  return null;
};

const FloorPlanEditorPage = ({
  fileName,
  imageUrl,
  initialBoxes,
  initialLines,
  onCancel,
  onSubmit,
  isSaving,
  onBoxesChange,
  onLinesChange,
  errorMessage,
}) => {
  const [boxes, setBoxes] = useState(initialBoxes);
  const [lines, setLines] = useState(initialLines);
  const [selectedItem, setSelectedItem] = useState(() => createSelectionFromData(initialBoxes, initialLines));
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
    setSelectedItem((prev) => {
      if (!prev) {
        return createSelectionFromData(initialBoxes, initialLines);
      }

      if (prev.type === 'box' && initialBoxes.some((box) => box.id === prev.id)) {
        return prev;
      }
      if (prev.type === 'line' && initialLines.some((line) => line.id === prev.id)) {
        return prev;
      }
      return createSelectionFromData(initialBoxes, initialLines);
    });
  }, [initialBoxes, initialLines]);

  const boxesMemo = useMemo(() => boxes, [boxes]);
  const linesMemo = useMemo(() => lines, [lines]);

  const handleUpdateBoxes = (producer) => {
    setBoxes((prev) => {
      const next = producer(prev);
      onBoxesChange?.(next);
      return next;
    });
  };

  const handleUpdateLines = (producer) => {
    setLines((prev) => {
      const next = producer(prev);
      onLinesChange?.(next);
      return next;
    });
  };

  const handleSelect = (item) => {
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
    } else {
      const nextId = `box-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      const prepared = { ...draft, id: nextId, labelId: draft.labelId ?? activeLabelId };
      handleUpdateBoxes((prev) => [...prev, prepared]);
      handleUpdateLines((prev) => subtractBoxFromLines(prev, prepared));
      setSelectedItem({ type: 'box', id: nextId });
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
        if (selectedItem?.type === 'line' && selectedItem.id === item.id) {
          const fallback = createSelectionFromData(boxesMemo, filtered);
          setSelectedItem(fallback);
        }
        return filtered;
      });
    } else {
      handleUpdateBoxes((prev) => {
        const filtered = prev.filter((box) => box.id !== item.id);
        if (selectedItem?.type === 'box' && selectedItem.id === item.id) {
          const fallback = createSelectionFromData(filtered, linesMemo);
          setSelectedItem(fallback);
        }
        return filtered;
      });
    }
  };

  const handleLabelChange = (item, labelId) => {
    if (!item) {
      return;
    }

    if (item.type === 'line') {
      handleUpdateLines((prev) => prev.map((line) => (line.id === item.id ? { ...line, labelId } : line)));
    } else {
      handleUpdateBoxes((prev) => prev.map((box) => (box.id === item.id ? { ...box, labelId } : box)));
    }
  };

  const handleSubmit = () => {
    onSubmit?.({ boxes: boxesMemo, lines: linesMemo });
  };

  const handleToggleAddMode = () => {
    setAddMode((prev) => !prev);
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
          selectedBox={selectedBox}
          selectedLine={selectedLine}
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
        />
      </main>
    </div>
  );
};

FloorPlanEditorPage.propTypes = {
  fileName: PropTypes.string,
  imageUrl: PropTypes.string.isRequired,
  initialBoxes: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      labelId: PropTypes.string.isRequired,
      x: PropTypes.number.isRequired,
      y: PropTypes.number.isRequired,
      width: PropTypes.number.isRequired,
      height: PropTypes.number.isRequired,
    })
  ),
  initialLines: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      labelId: PropTypes.string.isRequired,
      x1: PropTypes.number.isRequired,
      y1: PropTypes.number.isRequired,
      x2: PropTypes.number.isRequired,
      y2: PropTypes.number.isRequired,
    })
  ),
  onCancel: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
  isSaving: PropTypes.bool,
  onBoxesChange: PropTypes.func,
  onLinesChange: PropTypes.func,
  errorMessage: PropTypes.string,
};

FloorPlanEditorPage.defaultProps = {
  fileName: 'floor-plan.png',
  initialBoxes: [],
  initialLines: [],
  isSaving: false,
  onBoxesChange: undefined,
  onLinesChange: undefined,
  errorMessage: null,
};

export default FloorPlanEditorPage;
