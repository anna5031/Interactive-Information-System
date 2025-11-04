import PropTypes from 'prop-types';
import { useMemo } from 'react';
import styles from './StepTwoCanvas.module.css';

const NODE_COLORS = {
  corridor: '#38bdf8',
  room: '#4ade80',
  door: '#fb7185',
  door_endpoints: '#fbbf24',
  default: '#64748b',
};

const StepTwoCanvas = ({
  imageSize,
  graph,
  rooms,
  doors,
  wallLines,
  stairs,
  elevators,
  selectedEntity,
  onSelectEntity,
  isLoading,
  error,
  showRoomLabels,
  showDoorLabels,
}) => {
  const width = Math.max(imageSize?.width ?? 1000, 1);
  const height = Math.max(imageSize?.height ?? 1000, 1);

  const nodeMap = useMemo(() => {
    const map = new Map();
    (graph?.nodes ?? []).forEach((node) => {
      if (Array.isArray(node?.pos) && node.pos.length === 2) {
        const [x, y] = node.pos;
        map.set(node.id, {
          ...node,
          x: Number(x),
          y: Number(y),
        });
      }
    });
    return map;
  }, [graph]);

  const selectedRoomId = selectedEntity?.type === 'room' ? selectedEntity.nodeId : null;
  const selectedDoorId = selectedEntity?.type === 'door' ? selectedEntity.nodeId : null;

  const selectedRoomNodeIds = useMemo(() => {
    if (!selectedRoomId) {
      return new Set();
    }
    return new Set([selectedRoomId]);
  }, [selectedRoomId]);

  const selectedDoorNodeIds = useMemo(() => {
    const set = new Set();
    if (!selectedDoorId) {
      return set;
    }
    set.add(selectedDoorId);
    (graph?.edges ?? []).forEach((edge) => {
      if (edge.source === selectedDoorId && edge.target.startsWith('door_endpoints')) {
        set.add(edge.target);
      }
      if (edge.target === selectedDoorId && edge.source.startsWith('door_endpoints')) {
        set.add(edge.source);
      }
    });
    return set;
  }, [selectedDoorId, graph]);

  const edgeSegments = useMemo(() => {
    if (!graph?.edges?.length) {
      return [];
    }
    return graph.edges
      .map((edge, index) => {
        const source = nodeMap.get(edge.source);
        const target = nodeMap.get(edge.target);
        if (!source || !target) {
          return null;
        }
        const connectsRoomDoor =
          (edge.source.startsWith('room_') &&
            (edge.target.startsWith('door_') || edge.target.startsWith('door_endpoints')))
          ||
          (edge.target.startsWith('room_') &&
            (edge.source.startsWith('door_') || edge.source.startsWith('door_endpoints')));
        if (connectsRoomDoor) {
          return null;
        }
        const isActive =
          selectedRoomNodeIds.has(edge.source) ||
          selectedRoomNodeIds.has(edge.target) ||
          selectedDoorNodeIds.has(edge.source) ||
          selectedDoorNodeIds.has(edge.target);
        return {
          key: `${edge.source}-${edge.target}-${index}`,
          x1: source.x,
          y1: source.y,
          x2: target.x,
          y2: target.y,
          active: isActive,
        };
      })
      .filter(Boolean);
  }, [graph, nodeMap, selectedRoomNodeIds, selectedDoorNodeIds]);

  const nodeList = useMemo(() => Array.from(nodeMap.values()), [nodeMap]);

  const handleCanvasBackgroundClick = (event) => {
    if (event.target === event.currentTarget) {
      onSelectEntity?.(null);
    }
  };

  const handleRoomClick = (nodeId) => {
    onSelectEntity?.({ type: 'room', nodeId });
  };

  const handleDoorClick = (nodeId) => {
    onSelectEntity?.({ type: 'door', nodeId });
  };

  const handleNodeClick = (node) => {
    if (node.id.startsWith('room_')) {
      onSelectEntity?.({ type: 'room', nodeId: node.id });
    } else if (node.id.startsWith('door_')) {
      onSelectEntity?.({ type: 'door', nodeId: node.id });
    }
  };

  if (isLoading) {
    return (
      <div className={styles.canvasWrapper}>
        <div className={styles.stateMessage}>그래프 데이터를 불러오는 중...</div>
      </div>
    );
  }

  return (
    <div className={styles.canvasWrapper}>
      {error && <div className={styles.errorBanner}>{error}</div>}
      <svg
        className={styles.svgCanvas}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio='xMidYMid meet'
        onClick={handleCanvasBackgroundClick}
      >
        <rect x='0' y='0' width={width} height={height} className={styles.canvasBackground} />

        {wallLines.map((wall) => (
          <line
            key={wall.id}
            x1={wall.x1}
            y1={wall.y1}
            x2={wall.x2}
            y2={wall.y2}
            className={styles.wallLine}
          />
        ))}

        {rooms.map((room) => {
          if (!room.polygon) {
            return null;
          }
          const isSelected = selectedRoomId === room.nodeId;
          return (
            <g key={room.nodeId} className={styles.roomGroup}>
              <polygon
                points={room.polygon}
                className={`${styles.roomPolygon} ${isSelected ? styles.roomPolygonSelected : ''}`}
                onClick={(event) => {
                  event.stopPropagation();
                  handleRoomClick(room.nodeId);
                }}
              />
              {showRoomLabels && room.label && room.labelPosition && (
                <text
                  x={room.labelPosition[0]}
                  y={room.labelPosition[1]}
                  className={styles.roomLabel}
                >
                  {room.label}
                </text>
              )}
            </g>
          );
        })}

        {stairs.map((stair) => (
          <polygon key={stair.id} points={stair.polygon} className={styles.stairPolygon} />
        ))}

        {elevators.map((elevator) => (
          <polygon key={elevator.id} points={elevator.polygon} className={styles.elevatorPolygon} />
        ))}

        {edgeSegments.map((edge) => (
          <line
            key={edge.key}
            x1={edge.x1}
            y1={edge.y1}
            x2={edge.x2}
            y2={edge.y2}
            className={`${styles.graphEdge} ${edge.active ? styles.graphEdgeActive : ''}`}
          />
        ))}

        {doors.map((door) => {
          const isSelected = selectedDoorId === door.nodeId;
          if (door.polygon) {
            const centroid = door.centroid ?? null;
            return (
              <g key={door.nodeId}>
                <polygon
                  points={door.polygon}
                  className={`${styles.doorPolygon} ${isSelected ? styles.doorPolygonSelected : ''}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    handleDoorClick(door.nodeId);
                  }}
                />
                {showDoorLabels && door.label && centroid && (
                  <text
                    x={centroid[0]}
                    y={centroid[1] - 8}
                    className={styles.doorLabel}
                    onClick={(event) => {
                      event.stopPropagation();
                      handleDoorClick(door.nodeId);
                    }}
                  >
                    {door.label}
                  </text>
                )}
              </g>
            );
          }
          if (door.centroid) {
            const [cx, cy] = door.centroid;
            const radius = 4.6;
            return (
              <g key={door.nodeId}>
                <circle
                  cx={cx}
                  cy={cy}
                  r={radius}
                  className={`${styles.doorCircle} ${isSelected ? styles.doorCircleSelected : ''}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    handleDoorClick(door.nodeId);
                  }}
                />
                {showDoorLabels && door.label && (
                  <text
                    x={cx}
                    y={cy - radius - 6}
                    className={styles.doorLabel}
                    onClick={(event) => {
                      event.stopPropagation();
                      handleDoorClick(door.nodeId);
                    }}
                  >
                    {door.label}
                  </text>
                )}
              </g>
            );
          }
          return null;
        })}

        {nodeList.map((node) => {
          if (node.id.startsWith('room_') || node.id.startsWith('door_')) {
            return null;
          }
          const color =
            NODE_COLORS[node.type] ??
            (node.id.startsWith('door_endpoints') ? NODE_COLORS.door_endpoints : NODE_COLORS.default);
          const radius =
            node.type === 'corridor'
              ? 2.4
              : node.type === 'door'
              ? 4.4
              : node.type === 'room'
              ? 3.6
              : node.id.startsWith('door_endpoints')
              ? 2.2
              : 2.6;
          const isSelected =
            selectedRoomNodeIds.has(node.id) || selectedDoorNodeIds.has(node.id);
          const isSelectable = false;
          return (
            <circle
              key={node.id}
              cx={node.x}
              cy={node.y}
            r={radius}
            className={`${styles.graphNode} ${isSelected ? styles.graphNodeSelected : ''} ${
              isSelectable ? styles.graphNodeSelectable : ''
            }`}
            style={{ fill: color }}
            onClick={(event) => {
                if (!isSelectable) {
                  return;
                }
                event.stopPropagation();
                handleNodeClick(node);
              }}
            />
          );
        })}
      </svg>
    </div>
  );
};

StepTwoCanvas.propTypes = {
  imageSize: PropTypes.shape({
    width: PropTypes.number,
    height: PropTypes.number,
  }),
  graph: PropTypes.shape({
    nodes: PropTypes.array,
    edges: PropTypes.array,
  }),
  rooms: PropTypes.arrayOf(
    PropTypes.shape({
      nodeId: PropTypes.string.isRequired,
      polygon: PropTypes.string,
      label: PropTypes.string,
      labelPosition: PropTypes.arrayOf(PropTypes.number),
    })
  ),
  doors: PropTypes.arrayOf(
    PropTypes.shape({
      nodeId: PropTypes.string.isRequired,
      polygon: PropTypes.string,
      centroid: PropTypes.arrayOf(PropTypes.number),
      label: PropTypes.string,
    })
  ),
  wallLines: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      x1: PropTypes.number.isRequired,
      y1: PropTypes.number.isRequired,
      x2: PropTypes.number.isRequired,
      y2: PropTypes.number.isRequired,
    })
  ),
  stairs: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      polygon: PropTypes.string.isRequired,
    })
  ),
  elevators: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      polygon: PropTypes.string.isRequired,
    })
  ),
  selectedEntity: PropTypes.shape({
    type: PropTypes.oneOf(['room', 'door']),
    nodeId: PropTypes.string,
  }),
  onSelectEntity: PropTypes.func,
  isLoading: PropTypes.bool,
  error: PropTypes.string,
  showRoomLabels: PropTypes.bool,
  showDoorLabels: PropTypes.bool,
};

StepTwoCanvas.defaultProps = {
  imageSize: { width: 1000, height: 1000 },
  graph: null,
  rooms: [],
  doors: [],
  wallLines: [],
  stairs: [],
  elevators: [],
  selectedEntity: null,
  onSelectEntity: () => {},
  isLoading: false,
  error: null,
  showRoomLabels: true,
  showDoorLabels: true,
};

export default StepTwoCanvas;
