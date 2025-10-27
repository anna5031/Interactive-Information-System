import PropTypes from 'prop-types';
import styles from './VoiceBubble.module.css';

const variantMap = {
  prompt: 'speaking',
  listening: 'listening',
  thinking: 'thinking',
  speaking: 'speaking',
  awaiting_listening: 'idle',
  idle: 'idle',
};

function toVariantClass(variant) {
  switch (variant) {
    case 'listening':
      return styles.figureListening;
    case 'thinking':
      return styles.figureThinking;
    case 'speaking':
      return styles.figureSpeaking;
    default:
      return styles.figureIdle;
  }
}

function VoiceBubble({ status = 'idle', message = '' }) {
  const variant = variantMap[status] ?? 'idle';
  const figureClass = toVariantClass(variant);
  const isThinking = variant === 'thinking';

  return (
    <div className={styles.container}>
      <div className={`${styles.figure} ${figureClass}`} data-variant={variant}>
        <span className={`${styles.circle} ${styles.circleOne}`} />
        <span className={`${styles.circle} ${styles.circleTwo}`} />
        <span className={`${styles.circle} ${styles.circleThree}`} />
        <span className={`${styles.circle} ${styles.circleFour}`} />
        <span className={`${styles.circle} ${styles.circleFive}`} />
        {isThinking ? (
          <>
            <span className={`${styles.tailDot} ${styles.tailDotLarge}`} />
            <span className={`${styles.tailDot} ${styles.tailDotSmall}`} />
          </>
        ) : (
          <span className={`${styles.circle} ${styles.circleTail}`} />
        )}
      </div>
      {message ? <p className={styles.message}>{message}</p> : null}
    </div>
  );
}

VoiceBubble.propTypes = {
  status: PropTypes.string,
  message: PropTypes.string,
};

export default VoiceBubble;
