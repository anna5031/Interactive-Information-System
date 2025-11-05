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
const HALF_BORDER = BORDER_WIDTH / 2;
const HIGHLIGHT_COLOR = '#f97316';

const BoxAnnotation = ({
  box,
  label,
  isSelected,
  highlightEdges,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onResizePointerDown,
  onResizePointerMove,
}) => {
  const baseColor = label?.color || '#f59e0b';
  const highlightSet = highlightEdges ? new Set(highlightEdges) : new Set();

  const edgeColor = (edge) => (highlightSet.has(edge) ? HIGHLIGHT_COLOR : baseColor);

  const cornerStyle = (corner) => {
    switch (corner) {
      case 'top-left':
        return {
          top: 0,
          left: 0,
          transform: `translate(-50%, -50%) translate(${HALF_BORDER}px, ${HALF_BORDER}px)`,
        };
      case 'top-right':
        return {
          top: 0,
          left: '100%',
          transform: `translate(-50%, -50%) translate(-${HALF_BORDER}px, ${HALF_BORDER}px)`,
        };
      case 'bottom-left':
        return {
          top: '100%',
          left: 0,
          transform: `translate(-50%, -50%) translate(${HALF_BORDER}px, -${HALF_BORDER}px)`,
        };
      case 'bottom-right':
      default:
        return {
          top: '100%',
          left: '100%',
          transform: `translate(-50%, -50%) translate(-${HALF_BORDER}px, -${HALF_BORDER}px)`,
        };
    }
  };

  const edgeStyle = (edge) => {
    switch (edge) {
      case 'top':
        return {
          top: 0,
          left: '50%',
          transform: `translate(-50%, -50%) translate(0, ${HALF_BORDER}px)`,
        };
      case 'bottom':
        return {
          top: '100%',
          left: '50%',
          transform: `translate(-50%, -50%) translate(0, -${HALF_BORDER}px)`,
        };
      case 'left':
        return {
          top: '50%',
          left: 0,
          transform: `translate(-50%, -50%) translate(${HALF_BORDER}px, 0)`,
        };
      case 'right':
      default:
        return {
          top: '50%',
          left: '100%',
          transform: `translate(-50%, -50%) translate(-${HALF_BORDER}px, 0)`,
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
        borderWidth: BORDER_WIDTH,
        borderStyle: 'solid',
        borderTopColor: edgeColor('top'),
        borderRightColor: edgeColor('right'),
        borderBottomColor: edgeColor('bottom'),
        borderLeftColor: edgeColor('left'),
      }}
      onPointerDown={(event) => onPointerDown(event, box)}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      <span className={styles.label} style={{ backgroundColor: baseColor }}>
        {label?.name || box.labelId}
      </span>
      {isSelected &&
        CORNERS.map((corner) => {
          const [vertical, horizontal] = corner.split('-');
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
  highlightEdges: PropTypes.oneOfType([PropTypes.instanceOf(Set), PropTypes.arrayOf(PropTypes.string)]),
  onPointerDown: PropTypes.func.isRequired,
  onPointerMove: PropTypes.func.isRequired,
  onPointerUp: PropTypes.func.isRequired,
  onResizePointerDown: PropTypes.func.isRequired,
  onResizePointerMove: PropTypes.func.isRequired,
};

BoxAnnotation.defaultProps = {
  label: undefined,
  highlightEdges: undefined,
};

export default BoxAnnotation;
