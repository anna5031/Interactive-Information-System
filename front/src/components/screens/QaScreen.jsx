import { useAppState } from '../../state/AppStateContext';
import layout from '../../styles/ScreenLayout.module.css';
import styles from './QaScreen.module.css';
import VoiceBubble from '../qa/VoiceBubble';

function QaScreen() {
  const { qaState } = useAppState();
  const { status, displayMessage } = qaState;

  const bubbleStatusMap = {
    prompt: 'speaking',
    listening: 'listening',
    thinking: 'thinking',
    speaking: 'speaking',
    awaiting_listening: 'idle',
    idle: 'idle',
  };

  const bubbleStatus = bubbleStatusMap[status] ?? bubbleStatusMap.idle;

  return (
    <div className={layout.screen}>
      <div className={`${layout.content} ${styles.qaContent}`}>
        <VoiceBubble status={bubbleStatus} />
        <p className={styles.promptBody}>{displayMessage}</p>
      </div>
    </div>
  );
}

export default QaScreen;
