import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Clipboard, Download, Plus, Save, Trash2 } from 'lucide-react';
import PropTypes from 'prop-types';
import StepTwoCanvas from '../components/stepTwo/StepTwoCanvas';
import { getStoredStepOneResultById } from '../api/stepOneResults';
import { saveStepTwoResult } from '../api/stepTwoResults';
import { downloadJson } from '../utils/download';
import styles from './AdminStepTwoPage.module.css';

const buildRoomDisplayLabel = (name, number) => {
  const parts = [name, number].map((value) => value?.trim()).filter(Boolean);
  return parts.length > 0 ? parts.join(', ') : '';
};

const sanitizeEntries = (entries) => {
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries
    .map((entry) => ({
      key: entry?.key?.trim() ?? '',
      value: entry?.value?.trim() ?? '',
    }))
    .filter((entry) => entry.key || entry.value);
};

const copyText = async (text) => {
  if (!text) {
    // eslint-disable-next-line no-alert
    alert('복사할 내용이 없습니다.');
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    // eslint-disable-next-line no-alert
    alert('JSON이 복사되었습니다.');
  } catch (error) {
    console.error('Failed to copy', error);
    // eslint-disable-next-line no-alert
    alert('복사에 실패했습니다.');
  }
};

const KeyValueEditor = ({ entries, onChange, addButtonLabel }) => {
  const safeEntries = Array.isArray(entries) ? entries : [];

  const handleChange = (index, field, value) => {
    onChange(
      safeEntries.map((entry, entryIndex) =>
        entryIndex === index ? { ...entry, [field]: value } : entry
      )
    );
  };

  const handleAdd = () => {
    onChange([...safeEntries, { key: '', value: '' }]);
  };

  const handleRemove = (index) => {
    onChange(safeEntries.filter((_, entryIndex) => entryIndex !== index));
  };

  return (
    <div className={styles.keyValueEditor}>
      {safeEntries.length === 0 && <p className={styles.keyValueHint}>추가 정보를 입력해 주세요.</p>}
      {safeEntries.map((entry, index) => (
        <div key={`entry-${index}`} className={styles.keyValueRow}>
          <input
            type='text'
            placeholder='key (예: 용도)'
            value={entry.key}
            onChange={(event) => handleChange(index, 'key', event.target.value)}
          />
          <input
            type='text'
            placeholder='value (예: 세미나실)'
            value={entry.value}
            onChange={(event) => handleChange(index, 'value', event.target.value)}
          />
          <button type='button' className={styles.iconButton} onClick={() => handleRemove(index)}>
            <Trash2 size={16} />
          </button>
        </div>
      ))}
      <button type='button' className={styles.addButton} onClick={handleAdd}>
        <Plus size={16} />
        {addButtonLabel}
      </button>
    </div>
  );
};

KeyValueEditor.propTypes = {
  entries: PropTypes.arrayOf(
    PropTypes.shape({
      key: PropTypes.string,
      value: PropTypes.string,
    })
  ),
  onChange: PropTypes.func.isRequired,
  addButtonLabel: PropTypes.string,
};

KeyValueEditor.defaultProps = {
  entries: [],
  addButtonLabel: '필드 추가',
};

const AdminStepTwoPage = () => {
  const { stepOneId } = useParams();
  const navigate = useNavigate();

  const [stepOneResult, setStepOneResult] = useState(null);
  const [stepOneBoxes, setStepOneBoxes] = useState([]);
  const [stepOneLines, setStepOneLines] = useState([]);
  const [stepOnePoints, setStepOnePoints] = useState([]);
  const [roomsState, setRoomsState] = useState([]);
  const [doorsState, setDoorsState] = useState([]);
  const [stage, setStage] = useState('base');
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [savedResult, setSavedResult] = useState(null);
  const [selectedEntity, setSelectedEntity] = useState(null);

  useEffect(() => {
    const stored = stepOneId ? getStoredStepOneResultById(stepOneId) : null;
    if (!stored) {
      navigate('/admin/upload', { replace: true });
      return;
    }
    setStepOneResult(stored);

    const boxes = Array.isArray(stored?.yolo?.boxes) ? stored.yolo.boxes : [];
    const lines = Array.isArray(stored?.wall?.lines) ? stored.wall.lines : [];
    const points = Array.isArray(stored?.door?.points) ? stored.door.points : [];

    setStepOneBoxes(boxes);
    setStepOneLines(lines);
    setStepOnePoints(points);

    const initialRooms = boxes.filter((box) => String(box.labelId) === '2');
    setRoomsState(
      initialRooms.map((box) => ({
        nodeId: box.id,
        name: '',
        number: '',
        extra: [],
      }))
    );

    const initialDoors = points.filter((point) => String(point.labelId) === '0');
    setDoorsState(
      initialDoors.map((point) => ({
        nodeId: point.id,
        type: '',
        customType: '',
        extra: [],
      }))
    );

    setStage('base');
    setSavedResult(null);
    setSelectedEntity(null);
  }, [stepOneId, navigate]);

  const roomEntities = useMemo(
    () =>
      stepOneBoxes
        .filter((box) => String(box.labelId) === '2')
        .map((box, index) => ({
          nodeId: box.id,
          index: index + 1,
          box,
        })),
    [stepOneBoxes]
  );

  const doorEntities = useMemo(
    () =>
      stepOnePoints
        .filter((point) => String(point.labelId) === '0')
        .map((point, index) => ({
          nodeId: point.id,
          index: index + 1,
          point,
        })),
    [stepOnePoints]
  );

  const roomEntityMap = useMemo(() => new Map(roomEntities.map((entity) => [entity.nodeId, entity])), [roomEntities]);
  const doorEntityMap = useMemo(() => new Map(doorEntities.map((entity) => [entity.nodeId, entity])), [doorEntities]);

  useEffect(() => {
    if (!selectedEntity) {
      if (roomEntities.length > 0) {
        setSelectedEntity({ type: 'room', nodeId: roomEntities[0].nodeId });
      } else if (doorEntities.length > 0) {
        setSelectedEntity({ type: 'door', nodeId: doorEntities[0].nodeId });
      }
      return;
    }

    if (selectedEntity.type === 'room' && !roomEntityMap.has(selectedEntity.nodeId)) {
      setSelectedEntity(null);
    } else if (selectedEntity.type === 'door' && !doorEntityMap.has(selectedEntity.nodeId)) {
      setSelectedEntity(null);
    }
  }, [selectedEntity, roomEntities, doorEntities, roomEntityMap, doorEntityMap]);

  const handleSelectEntity = (entity) => {
    setSelectedEntity(entity);
    if (!entity) {
      return;
    }

    const selector = entity.type === 'room' ? `details[data-room-id="${entity.nodeId}"]` : `details[data-door-id="${entity.nodeId}"]`;
    window.requestAnimationFrame(() => {
      const detailsEl = document.querySelector(selector);
      if (detailsEl && !detailsEl.open) {
        detailsEl.open = true;
      }
    });
  };

  const handleRoomFieldChange = (nodeId, field, value) => {
    setRoomsState((prev) =>
      prev.map((room) => (room.nodeId === nodeId ? { ...room, [field]: value } : room))
    );
    setSelectedEntity({ type: 'room', nodeId });
  };

  const handleDoorFieldChange = (nodeId, field, value) => {
    setDoorsState((prev) =>
      prev.map((door) => (door.nodeId === nodeId ? { ...door, [field]: value } : door))
    );
    setSelectedEntity({ type: 'door', nodeId });
  };

  const handleRoomExtraChange = (nodeId, extra) => {
    setRoomsState((prev) =>
      prev.map((room) => (room.nodeId === nodeId ? { ...room, extra } : room))
    );
    setSelectedEntity({ type: 'room', nodeId });
  };

  const handleDoorExtraChange = (nodeId, extra) => {
    setDoorsState((prev) =>
      prev.map((door) => (door.nodeId === nodeId ? { ...door, extra } : door))
    );
    setSelectedEntity({ type: 'door', nodeId });
  };

  const goToDetailsStage = () => {
    const invalidRoom = roomsState.find((room) => !room.name?.trim() && !room.number?.trim());
    if (invalidRoom) {
      setSelectedEntity({ type: 'room', nodeId: invalidRoom.nodeId });
      // eslint-disable-next-line no-alert
      alert('각 방은 “방 이름” 또는 “방 호수” 중 하나 이상을 입력해야 합니다.');
      return;
    }

    setRoomsState((prev) => prev.map((room) => ({ ...room, extra: Array.isArray(room.extra) ? room.extra : [] })));
    setDoorsState((prev) => prev.map((door) => ({ ...door, extra: Array.isArray(door.extra) ? door.extra : [] })));
    setStage('details');
  };

  const handleSave = async () => {
    if (!stepOneResult) {
      return;
    }

    setIsSaving(true);
    setSaveError(null);

    const roomsPayload = roomsState.map((room) => {
      const entity = roomEntityMap.get(room.nodeId);
      const displayLabel = buildRoomDisplayLabel(room.name, room.number);
      const extra = sanitizeEntries(room.extra);
      const base = {
        name: room.name?.trim() ?? '',
        number: room.number?.trim() ?? '',
        displayLabel,
      };
      const meta = [
        { key: 'name', value: base.name },
        { key: 'number', value: base.number },
        { key: 'displayLabel', value: displayLabel },
        ...extra,
      ];
      return {
        nodeId: room.nodeId,
        base,
        meta,
        geometry: entity?.box ?? null,
      };
    });

    const doorsPayload = doorsState.map((door) => {
      const entity = doorEntityMap.get(door.nodeId);
      const resolvedType = door.type === '기타' ? door.customType?.trim() ?? '' : door.type?.trim() ?? '';
      const extra = sanitizeEntries(door.extra);
      const base = {
        type: door.type?.trim() ?? '',
      };
      if (door.type === '기타' && door.customType?.trim()) {
        base.customType = door.customType.trim();
      }
      const meta = [
        { key: 'type', value: resolvedType },
        ...(door.type === '기타' && door.customType?.trim()
          ? [{ key: 'customType', value: door.customType.trim() }]
          : []),
        ...extra,
      ];
      return {
        nodeId: door.nodeId,
        base,
        meta,
        geometry: entity?.point ?? null,
      };
    });

    try {
      const payload = {
        sourceFloorplan: stepOneResult.id,
        rooms: roomsPayload,
        doors: doorsPayload,
        preview: {
          imageUrl: stepOneResult.metadata?.imageUrl ?? null,
        },
      };
      const record = await saveStepTwoResult(payload);
      setSavedResult(record);
      setStage('review');
    } catch (error) {
      console.error(error);
      setSaveError('저장 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.');
    } finally {
      setIsSaving(false);
    }
  };

  const reviewJson = useMemo(() => (savedResult ? JSON.stringify(savedResult, null, 2) : ''), [savedResult]);

  const handleDownloadReview = () => {
    if (!savedResult) {
      return;
    }
    downloadJson(savedResult.fileName ?? savedResult.id ?? 'step_two_result', savedResult);
  };

  const handleFinish = () => {
    navigate('/admin/upload');
  };

  if (!stepOneResult) {
    return null;
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <button type='button' className={styles.backButton} onClick={() => navigate('/admin/upload')}>
          <ArrowLeft size={18} />
          메인으로
        </button>
        <div className={styles.heading}>
          <h1>2단계: 그래프 노드 메타데이터 입력</h1>
          <p>도면 ID: {stepOneResult.fileName ?? stepOneResult.id}</p>
        </div>
        <div className={styles.stageIndicator}>
          <span className={stage === 'base' ? styles.activeStage : ''}>기본 정보</span>
          <span className={stage === 'details' ? styles.activeStage : ''}>상세 정보</span>
          <span className={stage === 'review' ? styles.activeStage : ''}>결과 확인</span>
        </div>
      </header>

      {stage !== 'review' && (
        <div className={styles.workspace}>
          <div className={styles.canvasColumn}>
            {stepOneResult.metadata?.imageUrl ? (
              <StepTwoCanvas
                imageUrl={stepOneResult.metadata.imageUrl}
                boxes={stepOneBoxes}
                lines={stepOneLines}
                points={stepOnePoints}
                selectedEntity={selectedEntity}
                onSelectEntity={handleSelectEntity}
              />
            ) : (
              <div className={styles.canvasPlaceholder}>
                <p>도면 이미지를 찾을 수 없습니다.</p>
                <p>1단계 결과를 다시 저장해 주세요.</p>
              </div>
            )}
          </div>

          <div className={styles.formColumn}>
            {stage === 'base' && (
              <>
                <section className={styles.section}>
                  <div className={styles.sectionHeader}>
                    <div>
                      <h2>Room 노드 기본 정보</h2>
                      <p>Room 박스 {roomEntities.length}개의 이름과 호수를 입력하세요.</p>
                    </div>
                  </div>
                  <div className={styles.entityList}>
                    {roomEntities.length === 0 && <p className={styles.emptyMessage}>등록된 Room 박스가 없습니다.</p>}
                    {roomsState.map((room) => {
                      const entity = roomEntityMap.get(room.nodeId);
                      const isActive = selectedEntity?.type === 'room' && selectedEntity.nodeId === room.nodeId;
                      return (
                        <div
                          key={room.nodeId}
                          className={`${styles.entityRow} ${isActive ? styles.entityRowActive : ''}`}
                          onMouseEnter={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                          onClick={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                        >
                          <div className={styles.nodeBadge}>
                            <strong>{entity?.index ?? '-'}</strong>
                            <span>{room.nodeId}</span>
                          </div>
                          <div className={styles.baseFields}>
                            <label>
                              <span>방 이름</span>
                              <input
                                type='text'
                                placeholder='예: 터만홀'
                                value={room.name}
                                onFocus={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                                onChange={(event) => handleRoomFieldChange(room.nodeId, 'name', event.target.value)}
                              />
                            </label>
                            <label>
                              <span>방 호수</span>
                              <input
                                type='text'
                                placeholder='예: 301'
                                value={room.number}
                                onFocus={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                                onChange={(event) => handleRoomFieldChange(room.nodeId, 'number', event.target.value)}
                              />
                            </label>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>

                <section className={styles.section}>
                  <div className={styles.sectionHeader}>
                    <div>
                      <h2>Door 노드 기본 정보</h2>
                      <p>문 종류를 선택하고 필요하면 직접 입력해 주세요.</p>
                    </div>
                  </div>
                  <div className={styles.entityList}>
                    {doorEntities.length === 0 && <p className={styles.emptyMessage}>등록된 Door 포인트가 없습니다.</p>}
                    {doorsState.map((door) => {
                      const entity = doorEntityMap.get(door.nodeId);
                      const isActive = selectedEntity?.type === 'door' && selectedEntity.nodeId === door.nodeId;
                      return (
                        <div
                          key={door.nodeId}
                          className={`${styles.entityRow} ${isActive ? styles.entityRowActive : ''}`}
                          onMouseEnter={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                          onClick={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                        >
                          <div className={styles.nodeBadge}>
                            <strong>{entity?.index ?? '-'}</strong>
                            <span>{door.nodeId}</span>
                          </div>
                          <div className={styles.baseFields}>
                            <label>
                              <span>문 종류</span>
                              <select
                                value={door.type}
                                onFocus={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                                onChange={(event) => handleDoorFieldChange(door.nodeId, 'type', event.target.value)}
                              >
                                <option value=''>선택하세요</option>
                                <option value='미닫이'>미닫이</option>
                                <option value='여닫이'>여닫이</option>
                                <option value='기타'>기타</option>
                              </select>
                            </label>
                            {door.type === '기타' && (
                              <label>
                                <span>직접 입력</span>
                                <input
                                  type='text'
                                  placeholder='예: 양개문'
                                  value={door.customType}
                                  onFocus={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                                  onChange={(event) => handleDoorFieldChange(door.nodeId, 'customType', event.target.value)}
                                />
                              </label>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>

                <footer className={styles.footer}>
                  <button type='button' className={styles.primaryButton} onClick={goToDetailsStage}>
                    다음 단계
                    <ArrowRight size={16} />
                  </button>
                </footer>
              </>
            )}

            {stage === 'details' && (
              <>
                <section className={styles.section}>
                  <div className={styles.sectionHeader}>
                    <div>
                      <h2>Room 상세 정보</h2>
                      <p>기본 정보를 확인하고 필요한 추가 필드를 입력하세요.</p>
                    </div>
                  </div>
                  <div className={styles.detailsList}>
                    {roomsState.map((room) => {
                      const displayLabel = buildRoomDisplayLabel(room.name, room.number);
                      const title = displayLabel || room.name || room.nodeId;
                      const subtitle = displayLabel ? room.nodeId : '기본 정보 미입력';
                      const isActive = selectedEntity?.type === 'room' && selectedEntity.nodeId === room.nodeId;
                      return (
                        <details
                          key={room.nodeId}
                          data-room-id={room.nodeId}
                          className={`${styles.detailsItem} ${isActive ? styles.entityRowActive : ''}`}
                          onMouseEnter={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                        >
                          <summary
                            className={styles.entitySummary}
                            onClick={() => setSelectedEntity({ type: 'room', nodeId: room.nodeId })}
                          >
                            <span>{title}</span>
                            <span className={styles.summaryNote}>{subtitle}</span>
                          </summary>
                          <div className={styles.detailBody}>
                            <KeyValueEditor
                              entries={room.extra}
                              onChange={(extra) => handleRoomExtraChange(room.nodeId, extra)}
                              addButtonLabel='추가 정보 필드'
                            />
                          </div>
                        </details>
                      );
                    })}
                  </div>
                </section>

                <section className={styles.section}>
                  <div className={styles.sectionHeader}>
                    <div>
                      <h2>Door 상세 정보</h2>
                      <p>문 종류를 확인하고 추가 정보를 입력하세요.</p>
                    </div>
                  </div>
                  <div className={styles.detailsList}>
                    {doorsState.map((door) => {
                      const resolvedType = door.type === '기타' ? door.customType : door.type;
                      const title = resolvedType || '문 정보 미입력';
                      const subtitle = door.type ? door.nodeId : '문 종류를 선택하세요';
                      const isActive = selectedEntity?.type === 'door' && selectedEntity.nodeId === door.nodeId;
                      return (
                        <details
                          key={door.nodeId}
                          data-door-id={door.nodeId}
                          className={`${styles.detailsItem} ${isActive ? styles.entityRowActive : ''}`}
                          onMouseEnter={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                        >
                          <summary
                            className={styles.entitySummary}
                            onClick={() => setSelectedEntity({ type: 'door', nodeId: door.nodeId })}
                          >
                            <span>{title}</span>
                            <span className={styles.summaryNote}>{subtitle}</span>
                          </summary>
                          <div className={styles.detailBody}>
                            <KeyValueEditor
                              entries={door.extra}
                              onChange={(extra) => handleDoorExtraChange(door.nodeId, extra)}
                              addButtonLabel='추가 정보 필드'
                            />
                          </div>
                        </details>
                      );
                    })}
                  </div>
                </section>

                {saveError && <p className={styles.errorMessage}>{saveError}</p>}

                <footer className={styles.footer}>
                  <button type='button' className={styles.secondaryButton} onClick={() => setStage('base')}>
                    이전 단계
                  </button>
                  <button
                    type='button'
                    className={styles.primaryButton}
                    onClick={handleSave}
                    disabled={isSaving}
                  >
                    {isSaving ? '저장 중...' : 'JSON 저장'}
                    <Save size={16} />
                  </button>
                </footer>
              </>
            )}
          </div>
        </div>
      )}

      {stage === 'review' && savedResult && (
        <div className={styles.reviewSection}>
          <div className={styles.reviewHeader}>
            <h2>2단계 결과 저장 완료</h2>
            <p>step_two_result 폴더에 저장되었으며 JSON을 다운로드하거나 복사할 수 있습니다.</p>
          </div>
          <div className={styles.reviewMeta}>
            <div>
              <span className={styles.metaLabel}>파일명</span>
              <span>{savedResult.fileName}</span>
            </div>
            <div>
              <span className={styles.metaLabel}>저장 시각</span>
              <span>{savedResult.createdAt}</span>
            </div>
            <div>
              <span className={styles.metaLabel}>Step 1 ID</span>
              <span>{savedResult.sourceFloorplan}</span>
            </div>
          </div>
          <div className={styles.reviewActions}>
            <button type='button' className={styles.primaryButton} onClick={handleDownloadReview}>
              <Download size={16} />
              JSON 다운로드
            </button>
            <button type='button' className={styles.secondaryButton} onClick={() => copyText(reviewJson)}>
              <Clipboard size={16} />
              JSON 복사
            </button>
          </div>
          <textarea className={styles.reviewTextarea} readOnly value={reviewJson} />
          <footer className={styles.footer}>
            <button type='button' className={styles.primaryButton} onClick={handleFinish}>
              메인으로 돌아가기
            </button>
          </footer>
        </div>
      )}
    </div>
  );
};

export default AdminStepTwoPage;
