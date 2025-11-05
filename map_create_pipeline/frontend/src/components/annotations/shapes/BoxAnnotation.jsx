import PropTypes from 'prop-types';
import styles from '../AnnotationCanvas.module.css';

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
const BORDER_WIDTH = 2;
const BORDER_OFFSET = BORDER_WIDTH / 2;
const BORDER_EXTEND = 1;
// const HIGHLIGHT_COLOR = '#f97316'; // *** 수정: 더 이상 여기서 사용하지 않음 ***

const BoxAnnotation = ({
  box,
  label,
  isSelected,
  // highlightEdges, // *** 수정: prop 제거 ***
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onResizePointerDown,
  onResizePointerMove,
}) => {
  const baseColor = label?.color || '#f59e0b';
  // const highlightSet = highlightEdges ? new Set(highlightEdges) : new Set(); // *** 수정: 제거 ***

  // const edgeColor = (edge) => (highlightSet.has(edge) ? HIGHLIGHT_COLOR : baseColor); // *** 수정: 제거 ***
  const edgeStrokeStyle = (edge) => {
    // const color = edgeColor(edge); // *** 수정: 제거 ***
    const color = baseColor; // *** 수정: 항상 baseColor 사용 ***

    switch (edge) {
      case 'left':
        return {
          top: `-${BORDER_EXTEND}px`,
          bottom: `-${BORDER_EXTEND}px`,
          left: 0,
          width: BORDER_WIDTH,
          transform: `translateX(-${BORDER_OFFSET}px)`,
          backgroundColor: color,
        };
      case 'right':
        return {
          top: `-${BORDER_EXTEND}px`,
          bottom: `-${BORDER_EXTEND}px`,
          left: '100%',
          width: BORDER_WIDTH,
          transform: `translateX(-${BORDER_OFFSET}px)`,
          backgroundColor: color,
        };
      case 'top':
        return {
          left: `-${BORDER_OFFSET}px`,
          right: `-${BORDER_OFFSET}px`,
          top: 0,
          height: BORDER_WIDTH,
          transform: `translateY(-${BORDER_OFFSET}px)`,
          backgroundColor: color,
        };
      case 'bottom':
      default:
        return {
          left: `-${BORDER_OFFSET}px`,
          right: `-${BORDER_OFFSET}px`,
          top: '100%',
          height: BORDER_WIDTH,
          transform: `translateY(-${BORDER_OFFSET}px)`,
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
      <span className={styles.label} style={{ backgroundColor: baseColor }}>
        {label?.name || box.labelId}
      </span>
      {isSelected &&
        CORNERS.map((corner) => {
          // const [vertical, horizontal] = corner.split('-');
          const cursor = CURSOR_MAP[corner] || 'pointer';
          const style = {
            ...cornerStyle(corner),
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
  // highlightEdges: PropTypes.oneOfType([PropTypes.instanceOf(Set), PropTypes.arrayOf(PropTypes.string)]), // *** 수정: prop 제거 ***
  onPointerDown: PropTypes.func.isRequired,
  onPointerMove: PropTypes.func.isRequired,
  onPointerUp: PropTypes.func.isRequired,
  onResizePointerDown: PropTypes.func.isRequired,
  onResizePointerMove: PropTypes.func.isRequired,
};

BoxAnnotation.defaultProps = {
  label: undefined,
  // highlightEdges: undefined, // *** 수정: prop 제거 ***
};

export default BoxAnnotation;
