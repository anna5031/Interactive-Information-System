import PropTypes from 'prop-types';
import styles from '../AnnotationCanvas.module.css';

const CORNERS = ['top-left', 'top-right', 'bottom-left', 'bottom-right'];
const CURSOR_MAP = {
  'top-left': 'nwse-resize',
  'top-right': 'nesw-resize',
  'bottom-left': 'nesw-resize',
  'bottom-right': 'nwse-resize',
};

const BoxAnnotation = ({
  box,
  label,
  isSelected,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onResizePointerDown,
  onResizePointerMove,
}) => {
  const borderColor = label?.color || '#f59e0b';

  return (
    <div
      role='presentation'
      className={`${styles.annotation} ${isSelected ? styles.selected : ''}`}
      style={{
        left: `${box.x * 100}%`,
        top: `${box.y * 100}%`,
        width: `${box.width * 100}%`,
        height: `${box.height * 100}%`,
        borderColor,
      }}
      onPointerDown={(event) => onPointerDown(event, box)}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      <span className={styles.label} style={{ backgroundColor: borderColor }}>
        {label?.name || box.labelId}
      </span>
      {isSelected &&
        CORNERS.map((corner) => {
          const [vertical, horizontal] = corner.split('-');
          const style = {
            top: vertical === 'top' ? 0 : undefined,
            bottom: vertical === 'bottom' ? 0 : undefined,
            left: horizontal === 'left' ? 0 : undefined,
            right: horizontal === 'right' ? 0 : undefined,
            cursor: CURSOR_MAP[corner] || 'pointer',
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
};

BoxAnnotation.defaultProps = {
  label: undefined,
};

export default BoxAnnotation;
