import PropTypes from 'prop-types';
import { useEffect, useMemo, useRef } from 'react';
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
  enableNodeDrag,
  onNodePositionChange,
  draggableNodeTypes,
  selectedGraphNodeIds = [],
  showStairLabels,
  showElevatorLabels,
  onSelectEdge,
}) => {
  const width = Math.max(imageSize?.width ?? 1000, 1);
  const height = Math.max(imageSize?.height ?? 1000, 1);
  const svgRef = useRef(null);
  const dragStateRef = useRef(null);

  const effectiveShowStairLabels =
    typeof showStairLabels === 'boolean' ? showStairLabels : showDoorLabels;
  const effectiveShowElevatorLabels =
    typeof showElevatorLabels === 'boolean' ? showElevatorLabels : showDoorLabels;

  useEffect(() => {
    return () => {
      if (dragStateRef.current) {
        window.removeEventListener('pointermove', dragStateRef.current.moveHandler);
        window.removeEventListener('pointerup', dragStateRef.current.upHandler);
      }
    };
  }, []);

  const clamp = (value, min, max) => {
    if (!Number.isFinite(value)) {
      return min;
    }
    return Math.min(Math.max(value, min), max);
  };

  const isNodeDraggable = (node) => {
    if (!enableNodeDrag) {
      return false;
    }
    if (!Array.isArray(node?.pos) || node.pos.length < 2) {
      return false;
    }
    if (typeof node?.id === 'string' && node.id.startsWith('door_')) {
      return false;
    }
    if (!draggableNodeTypes || draggableNodeTypes.length === 0) {
      return true;
    }
    const normalized = (node.type || '').toLowerCase();
    return draggableNodeTypes.includes(normalized);
  };

  const toCanvasCoordinates = (clientX, clientY) => {
    const svgElement = svgRef.current;
    if (!svgElement) {
      return null;
    }
    const rect = svgElement.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      return null;
    }
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  };

  const attachGlobalHandlers = () => {
    const moveHandler = (event) => {
      const dragState = dragStateRef.current;
      if (!dragState || !dragState.nodeId || !enableNodeDrag || !onNodePositionChange) {
        return;
      }
      const coords = toCanvasCoordinates(event.clientX, event.clientY);
      if (!coords) {
        return;
      }
      const newX = clamp(coords.x - dragState.offsetX, 0, width);
      const newY = clamp(coords.y - dragState.offsetY, 0, height);
      dragState.lastPosition = [newX, newY];
      onNodePositionChange(dragState.nodeId, [newX, newY]);
    };

    const upHandler = () => {
      const dragState = dragStateRef.current;
      dragStateRef.current = null;
      window.removeEventListener('pointermove', moveHandler);
      window.removeEventListener('pointerup', upHandler);
      if (dragState?.onComplete) {
        dragState.onComplete();
      }
    };

    window.addEventListener('pointermove', moveHandler);
    window.addEventListener('pointerup', upHandler);

    dragStateRef.current = {
      ...dragStateRef.current,
      moveHandler,
      upHandler,
    };
  };

  const handleNodePointerDown = (node, event) => {
    if (!isNodeDraggable(node) || typeof onNodePositionChange !== 'function') {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const svgElement = svgRef.current;
    if (!svgElement) {
      return;
    }
    const coords = toCanvasCoordinates(event.clientX, event.clientY);
    if (!coords) {
      return;
    }
    const offsetX = coords.x - Number(node.x ?? 0);
    const offsetY = coords.y - Number(node.y ?? 0);

    dragStateRef.current = {
      nodeId: node.id,
      offsetX,
      offsetY,
    };
    attachGlobalHandlers();
  };

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

  const doorEndpointsByDoor = useMemo(() => {
    const map = new Map();
    (graph?.nodes ?? []).forEach((node) => {
      if ((node?.type || '').toLowerCase() !== 'door_endpoints') {
        return;
      }
      const links = node?.attributes?.door_link_ids || node?.attributes?.doorLinkIds || [];
      links.forEach((rawId) => {
        const normalized =
          typeof rawId === 'number'
            ? `door_${rawId}`
            : String(rawId).startsWith('door_')
              ? String(rawId)
              : `door_${rawId}`;
        if (!map.has(normalized)) {
          map.set(normalized, new Set());
        }
        map.get(normalized).add(node.id);
      });
    });
    return map;
  }, [graph]);

  const selectedDoorNodeIds = useMemo(() => {
    const set = new Set();
    const doorIds = new Set(
      (selectedGraphNodeIds || []).filter((nodeId) => typeof nodeId === 'string' && nodeId.startsWith('door_'))
    );
    if (selectedDoorId) {
      doorIds.add(selectedDoorId);
    }
    doorIds.forEach((doorId) => {
      set.add(doorId);
      const endpoints = doorEndpointsByDoor.get(doorId);
      if (endpoints) {
        endpoints.forEach((endpointId) => set.add(endpointId));
      }
    });
    return set;
  }, [selectedDoorId, selectedGraphNodeIds, doorEndpointsByDoor]);

  const selectedGraphNodesSet = useMemo(
    () => new Set(selectedGraphNodeIds || []),
    [selectedGraphNodeIds]
  );

  const selectedPair =
    Array.isArray(selectedGraphNodeIds) && selectedGraphNodeIds.length === 2
      ? [selectedGraphNodeIds[0], selectedGraphNodeIds[1]]
      : null;

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
        const connectsSelectedPair =
          selectedPair &&
          ((edge.source === selectedPair[0] && edge.target === selectedPair[1]) ||
            (edge.source === selectedPair[1] && edge.target === selectedPair[0]));
        const isActive =
          selectedPair != null
            ? connectsSelectedPair
            : selectedRoomNodeIds.has(edge.source) ||
              selectedRoomNodeIds.has(edge.target) ||
              selectedDoorNodeIds.has(edge.source) ||
              selectedDoorNodeIds.has(edge.target) ||
              selectedGraphNodesSet.has(edge.source) ||
              selectedGraphNodesSet.has(edge.target);
        return {
          key: `${edge.source}-${edge.target}-${index}`,
          x1: source.x,
          y1: source.y,
          x2: target.x,
          y2: target.y,
          active: isActive,
          sourceId: edge.source,
          targetId: edge.target,
        };
      })
      .filter(Boolean);
  }, [graph, nodeMap, selectedRoomNodeIds, selectedDoorNodeIds, selectedGraphNodesSet, selectedPair]);

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

  const handleStairClick = (nodeId) => {
    if (!enableNodeDrag) {
      return;
    }
    onSelectEntity?.({ type: 'graph-node', nodeId });
  };

  const handleElevatorClick = (nodeId) => {
    if (!enableNodeDrag) {
      return;
    }
    onSelectEntity?.({ type: 'graph-node', nodeId });
  };

  const handleEdgeClick = (edge, event) => {
    if (!onSelectEdge) {
      return;
    }
    event.stopPropagation();
    if (!edge?.sourceId || !edge?.targetId) {
      return;
    }
    onSelectEdge({ source: edge.sourceId, target: edge.targetId });
  };

  const handleNodeClick = (node) => {
    if (node.id.startsWith('room_')) {
      onSelectEntity?.({ type: 'room', nodeId: node.id });
    } else if (node.id.startsWith('door_')) {
      onSelectEntity?.({ type: 'door', nodeId: node.id });
    } else {
      onSelectEntity?.({ type: 'graph-node', nodeId: node.id });
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
        ref={svgRef}
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

        {stairs.map((stair) => {
          const nodeId = stair.nodeId || stair.id;
          const isSelected = selectedGraphNodesSet.has(nodeId);
          const stairLabel = stair.label ?? stair.id;
          return (
            <g key={stair.id}>
              <polygon
                points={stair.polygon}
                className={`${styles.stairPolygon} ${isSelected ? styles.stairPolygonSelected : ''}`}
                onClick={(event) => {
                  if (!enableNodeDrag) {
                    return;
                  }
                  event.stopPropagation();
                  handleStairClick(nodeId);
                }}
              />
              {effectiveShowStairLabels && stairLabel && stair.centroid && (
                <text
                  x={stair.centroid[0]}
                  y={stair.centroid[1]}
                  className={styles.stairLabel}
                  onClick={(event) => {
                    if (!enableNodeDrag) {
                      return;
                    }
                    event.stopPropagation();
                    handleStairClick(nodeId);
                  }}
                >
                  {stairLabel}
                </text>
              )}
            </g>
          );
        })}

        {elevators.map((elevator) => {
          const nodeId = elevator.nodeId || elevator.id;
          const isSelected = selectedGraphNodesSet.has(nodeId);
          const elevatorLabel = elevator.label ?? elevator.id;
          return (
            <g key={elevator.id}>
              <polygon
                points={elevator.polygon}
                className={`${styles.elevatorPolygon} ${isSelected ? styles.elevatorPolygonSelected : ''}`}
                onClick={(event) => {
                  if (!enableNodeDrag) {
                    return;
                  }
                  event.stopPropagation();
                  handleElevatorClick(nodeId);
                }}
              />
              {effectiveShowElevatorLabels && elevatorLabel && elevator.centroid && (
                <text
                  x={elevator.centroid[0]}
                  y={elevator.centroid[1]}
                  className={styles.elevatorLabel}
                  onClick={(event) => {
                    if (!enableNodeDrag) {
                      return;
                    }
                    event.stopPropagation();
                    handleElevatorClick(nodeId);
                  }}
                >
                  {elevatorLabel}
                </text>
              )}
            </g>
          );
        })}

        {edgeSegments.map((edge) => (
          <line
            key={edge.key}
            x1={edge.x1}
            y1={edge.y1}
            x2={edge.x2}
            y2={edge.y2}
            className={`${styles.graphEdge} ${edge.active ? styles.graphEdgeActive : ''} ${
              onSelectEdge ? styles.graphEdgeSelectable : ''
            }`}
            onClick={(event) => handleEdgeClick(edge, event)}
          />
        ))}

        {doors.map((door) => {
          const isSelected = selectedDoorId === door.nodeId || selectedGraphNodesSet.has(door.nodeId);
          const labelNumber = door.index != null ? Math.max(0, door.index - 1) : null;
          const doorLabelText =
            labelNumber != null
              ? `door ${labelNumber}`
              : door.label ?? (door.nodeId?.replace(/^door_/, '') || door.nodeId);
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
                {showDoorLabels && doorLabelText && centroid && (
                  <text
                    x={centroid[0]}
                    y={centroid[1] - 8}
                    className={styles.doorLabel}
                    onClick={(event) => {
                      event.stopPropagation();
                      handleDoorClick(door.nodeId);
                    }}
                  >
                    {doorLabelText}
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
                {showDoorLabels && doorLabelText && (
                  <text
                    x={cx}
                    y={cy - radius - 6}
                    className={styles.doorLabel}
                    onClick={(event) => {
                      event.stopPropagation();
                      handleDoorClick(door.nodeId);
                    }}
                  >
                    {doorLabelText}
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
            selectedGraphNodesSet.has(node.id) ||
            selectedRoomNodeIds.has(node.id) ||
            selectedDoorNodeIds.has(node.id);
          const isSelectable = enableNodeDrag && isNodeDraggable(node);
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
              onPointerDown={(event) => handleNodePointerDown(node, event)}
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
      nodeId: PropTypes.string,
      polygon: PropTypes.string.isRequired,
      centroid: PropTypes.arrayOf(PropTypes.number),
      label: PropTypes.string,
    })
  ),
  elevators: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      nodeId: PropTypes.string,
      polygon: PropTypes.string.isRequired,
      centroid: PropTypes.arrayOf(PropTypes.number),
      label: PropTypes.string,
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
  enableNodeDrag: PropTypes.bool,
  onNodePositionChange: PropTypes.func,
  draggableNodeTypes: PropTypes.arrayOf(PropTypes.string),
  selectedGraphNodeIds: PropTypes.arrayOf(PropTypes.string),
  showStairLabels: PropTypes.bool,
  showElevatorLabels: PropTypes.bool,
  onSelectEdge: PropTypes.func,
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
  enableNodeDrag: false,
  onNodePositionChange: null,
  draggableNodeTypes: undefined,
  selectedGraphNodeIds: [],
  showStairLabels: undefined,
  showElevatorLabels: undefined,
  onSelectEdge: undefined,
};

export default StepTwoCanvas;
