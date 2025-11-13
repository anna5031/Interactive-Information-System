import PropTypes from 'prop-types';
import styles from '../AnnotationCanvas.module.css';

const PointAnnotation = ({
  point,
  label,
  isSelected,
  imageBox,
  onPointerDown,
  onPointerMove,
  onPointerUp,
}) => {
  const fill = label?.color || '#ef4444';
  const cx = point.x * imageBox.width;
  const cy = point.y * imageBox.height;

  return (
    <circle
      cx={cx}
      cy={cy}
      r={isSelected ? 7 : 5}
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
};

PointAnnotation.defaultProps = {
  label: undefined,
};

export default PointAnnotation;
