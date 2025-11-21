import PropTypes from 'prop-types';
import styles from '../AnnotationCanvas.module.css';

const DEFAULT_SCREEN_RADIUS = 5;
const SELECTED_SCREEN_RADIUS = 8;

const PointAnnotation = ({
  point,
  label,
  isSelected,
  imageBox,
  viewportScale,
  onPointerDown,
  onPointerMove,
  onPointerUp,
}) => {
  const fill = label?.color || '#ef4444';
  const cx = point.x * imageBox.width;
  const cy = point.y * imageBox.height;
  const scale = Number.isFinite(viewportScale) && viewportScale > 0 ? viewportScale : 1;
  const targetScreenRadius = isSelected ? SELECTED_SCREEN_RADIUS : DEFAULT_SCREEN_RADIUS;
  const radius = targetScreenRadius / scale;

  return (
    <circle
      cx={cx}
      cy={cy}
      r={radius}
      className={`${styles.point} ${isSelected ? styles.pointSelected : ''}`}
      fill={fill}
      onPointerDown={(event) => onPointerDown(event, point)}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    />
  );
};

PointAnnotation.propTypes = {
  point: PropTypes.shape({
    id: PropTypes.string.isRequired,
    labelId: PropTypes.string.isRequired,
    x: PropTypes.number.isRequired,
    y: PropTypes.number.isRequired,
  }).isRequired,
  label: PropTypes.shape({
    name: PropTypes.string,
    color: PropTypes.string,
  }),
  isSelected: PropTypes.bool.isRequired,
  imageBox: PropTypes.shape({
    width: PropTypes.number.isRequired,
    height: PropTypes.number.isRequired,
  }).isRequired,
  onPointerDown: PropTypes.func.isRequired,
  onPointerMove: PropTypes.func.isRequired,
  onPointerUp: PropTypes.func.isRequired,
  viewportScale: PropTypes.number,
};

PointAnnotation.defaultProps = {
  label: undefined,
  viewportScale: 1,
};

export default PointAnnotation;
