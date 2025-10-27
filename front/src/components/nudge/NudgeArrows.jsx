import PropTypes from 'prop-types';
import styles from './NudgeArrows.module.css';

function NudgeArrows({ count = 5 }) {
  const arrows = Array.from({ length: count }, (_, index) => index);

  return (
    <div className={styles.arrows}>
      {arrows.map((index) => (
        <span key={index} className={styles.arrow} style={{ '--delay': `${index * 0.15}s` }} />
      ))}
    </div>
  );
}

NudgeArrows.propTypes = {
  count: PropTypes.number,
};

export default NudgeArrows;
