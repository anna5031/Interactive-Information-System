import PropTypes from 'prop-types';
import { useMemo } from 'react';
import AnnotationCanvas from '../annotations/AnnotationCanvas';
import styles from './StepTwoCanvas.module.css';

const selectableBoxLabelIds = new Set(['2']);
const selectablePointLabelIds = new Set(['0']);

const StepTwoCanvas = ({ imageUrl, boxes, lines, points, selectedEntity, onSelectEntity }) => {
  const selectableBoxes = useMemo(
    () => boxes.filter((box) => selectableBoxLabelIds.has(String(box.labelId))),
    [boxes]
  );
  const selectablePoints = useMemo(
    () => points.filter((point) => selectablePointLabelIds.has(String(point.labelId))),
    [points]
  );

  const selectedItem = useMemo(() => {
    if (!selectedEntity) {
      return null;
    }
    if (selectedEntity.type === 'room') {
      return { type: 'box', id: selectedEntity.nodeId };
    }
    if (selectedEntity.type === 'door') {
      return { type: 'point', id: selectedEntity.nodeId };
    }
    return null;
  }, [selectedEntity]);

  const handleSelect = (item) => {
    if (!item) {
      onSelectEntity(null);
      return;
    }
    if (item.type === 'box' && selectableBoxes.some((box) => box.id === item.id)) {
      onSelectEntity({ type: 'room', nodeId: item.id });
      return;
    }
    if (item.type === 'point' && selectablePoints.some((point) => point.id === item.id)) {
      onSelectEntity({ type: 'door', nodeId: item.id });
      return;
    }
  };

  return (
    <div className={styles.canvasWrapper}>
      <AnnotationCanvas
        imageUrl={imageUrl}
        boxes={boxes}
        lines={lines}
        points={points}
        selectedItem={selectedItem}
        onSelect={handleSelect}
        addMode={false}
        activeLabelId='0'
        isReadOnly
      />
    </div>
  );
};

StepTwoCanvas.propTypes = {
  imageUrl: PropTypes.string.isRequired,
  boxes: PropTypes.arrayOf(PropTypes.object),
  lines: PropTypes.arrayOf(PropTypes.object),
  points: PropTypes.arrayOf(PropTypes.object),
  selectedEntity: PropTypes.shape({
    type: PropTypes.oneOf(['room', 'door']),
    nodeId: PropTypes.string,
  }),
  onSelectEntity: PropTypes.func,
};

StepTwoCanvas.defaultProps = {
  boxes: [],
  lines: [],
  points: [],
  selectedEntity: null,
  onSelectEntity: () => {},
};

export default StepTwoCanvas;
