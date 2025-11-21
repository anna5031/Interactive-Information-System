import PropTypes from 'prop-types';
import styles from '../AnnotationCanvas.module.css';
import { BOX_BORDER_WIDTH, BOX_BORDER_OFFSET, BOX_BORDER_EXTEND } from '../boxDrawingConstants';

const CORNERS = ['top-left', 'top-right', 'bottom-left', 'bottom-right'];
const CURSOR_MAP = {
  'top-left': 'nwse-resize',
  'top-right': 'nesw-resize',
  'bottom-left': 'nesw-resize',
  'bottom-right': 'nwse-resize',
};
const EDGES = ['top', 'right', 'bottom', 'left'];
const EDGE_CURSOR_MAP = {
  top: 'ns-resize',
  bottom: 'ns-resize',
  left: 'ew-resize',
  right: 'ew-resize',
};
const HANDLE_SCREEN_SIZE = 10;
const HANDLE_BORDER_WIDTH = 2;

const BoxAnnotation = ({
  box,
  label,
  isSelected,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onResizePointerDown,
  onResizePointerMove,
  viewportScale = 1,
}) => {
  const baseColor = label?.color || '#f59e0b';
  const safeScale = Number.isFinite(viewportScale) && viewportScale > 0 ? viewportScale : 1;
  const handleSize = HANDLE_SCREEN_SIZE / safeScale;
  const handleBorderWidth = HANDLE_BORDER_WIDTH / safeScale;
  const handleDimensions = {
    width: `${handleSize}px`,
    height: `${handleSize}px`,
    borderWidth: `${handleBorderWidth}px`,
  };
  const edgeStrokeStyle = (edge) => {
    const color = baseColor;
    const strokeThickness = `${BOX_BORDER_WIDTH}px`;

    switch (edge) {
      case 'left':
        return {
          top: `-${BOX_BORDER_EXTEND}px`,
          bottom: `-${BOX_BORDER_EXTEND}px`,
          left: 0,
          width: strokeThickness,
          transform: `translateX(-${BOX_BORDER_OFFSET}px)`,
          backgroundColor: color,
        };
      case 'right':
        return {
          top: `-${BOX_BORDER_EXTEND}px`,
          bottom: `-${BOX_BORDER_EXTEND}px`,
          left: '100%',
          width: strokeThickness,
          transform: `translateX(-${BOX_BORDER_OFFSET}px)`,
          backgroundColor: color,
        };
      case 'top':
        return {
          left: `-${BOX_BORDER_OFFSET}px`,
          right: `-${BOX_BORDER_OFFSET}px`,
          top: 0,
          height: strokeThickness,
          transform: `translateY(-${BOX_BORDER_OFFSET}px)`,
          backgroundColor: color,
        };
      case 'bottom':
      default:
        return {
          left: `-${BOX_BORDER_OFFSET}px`,
          right: `-${BOX_BORDER_OFFSET}px`,
          top: '100%',
          height: strokeThickness,
          transform: `translateY(-${BOX_BORDER_OFFSET}px)`,
          backgroundColor: color,
        };
    }
  };

  const cornerStyle = (corner) => {
    switch (corner) {
      case 'top-left':
        return {
          top: 0,
          left: 0,
          transform: 'translate(-50%, -50%)',
        };
      case 'top-right':
        return {
          top: 0,
          left: '100%',
          transform: 'translate(-50%, -50%)',
        };
      case 'bottom-left':
        return {
          top: '100%',
          left: 0,
          transform: 'translate(-50%, -50%)',
        };
      case 'bottom-right':
      default:
        return {
          top: '100%',
          left: '100%',
          transform: 'translate(-50%, -50%)',
        };
    }
  };

  const edgeStyle = (edge) => {
    switch (edge) {
      case 'top':
        return {
          top: 0,
          left: '50%',
          transform: 'translate(-50%, -50%)',
        };
      case 'bottom':
        return {
          top: '100%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
        };
      case 'left':
        return {
          top: '50%',
          left: 0,
          transform: 'translate(-50%, -50%)',
        };
      case 'right':
      default:
        return {
          top: '50%',
          left: '100%',
          transform: 'translate(-50%, -50%)',
        };
    }
  };

  return (
    <div
      role='presentation'
      className={`${styles.annotation} ${isSelected ? styles.selected : ''}`}
      style={{
        left: `${box.x * 100}%`,
        top: `${box.y * 100}%`,
        width: `${box.width * 100}%`,
        height: `${box.height * 100}%`,
      }}
      onPointerDown={(event) => onPointerDown(event, box)}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {EDGES.map((edge) => (
        <div
          key={`stroke-${edge}`}
          role='presentation'
          aria-hidden='true'
          className={styles.edgeStroke}
          style={edgeStrokeStyle(edge)}
        />
      ))}
      {isSelected &&
        CORNERS.map((corner) => {
          // const [vertical, horizontal] = corner.split('-');
          const cursor = CURSOR_MAP[corner] || 'pointer';
          const style = {
            ...cornerStyle(corner),
            ...handleDimensions,
            cursor,
          };

          return (
            <div
              key={corner}
              role='presentation'
              className={styles.resizeHandle}
              style={style}
              onPointerDown={(event) => onResizePointerDown(event, box, corner)}
              onPointerMove={onResizePointerMove}
              onPointerUp={onPointerUp}
            />
          );
        })}
      {isSelected &&
        EDGES.map((edge) => {
          const cursor = EDGE_CURSOR_MAP[edge] || 'pointer';
          const style = {
            ...edgeStyle(edge),
            ...handleDimensions,
            cursor,
          };

          return (
            <div
              key={`edge-${edge}`}
              role='presentation'
              className={`${styles.edgeHandle} ${
                edge === 'top' || edge === 'bottom' ? styles.edgeHandleHorizontal : styles.edgeHandleVertical
              }`}
              style={style}
              onPointerDown={(event) => onResizePointerDown(event, box, edge)}
              onPointerMove={onResizePointerMove}
              onPointerUp={onPointerUp}
            />
          );
        })}
    </div>
  );
};

BoxAnnotation.propTypes = {
  box: PropTypes.shape({
    id: PropTypes.string.isRequired,
    labelId: PropTypes.string.isRequired,
    x: PropTypes.number.isRequired,
    y: PropTypes.number.isRequired,
    width: PropTypes.number.isRequired,
    height: PropTypes.number.isRequired,
  }).isRequired,
  label: PropTypes.shape({
    name: PropTypes.string,
    color: PropTypes.string,
  }),
  isSelected: PropTypes.bool.isRequired,
  onPointerDown: PropTypes.func.isRequired,
  onPointerMove: PropTypes.func.isRequired,
  onPointerUp: PropTypes.func.isRequired,
  onResizePointerDown: PropTypes.func.isRequired,
  onResizePointerMove: PropTypes.func.isRequired,
  viewportScale: PropTypes.number,
};

BoxAnnotation.defaultProps = {
  label: undefined,
  viewportScale: 1,
};

export default BoxAnnotation;
