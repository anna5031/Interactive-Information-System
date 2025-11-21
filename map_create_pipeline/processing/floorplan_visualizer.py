from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import networkx as nx

EXPORT_IMAGE_FILENAME = "annotated_floorplan.png"


class FloorPlanVisualizer:
    """객체/그래프를 동시에 시각화해 디버깅과 보고에 활용하는 도구."""

    def __init__(self, graph: nx.Graph, objects: Dict[str, List[dict]]):
        """그래프와 객체 컬렉션을 받아 렌더링 준비를 한다."""
        self.G = graph
        self.objs = objects

    def show(
        self,
        path=None,
        title: str = "Floor Plan Navigation",
        output_path: Optional[Path] = None,
        show_window: bool = True,
        annotate_connections: bool = True,
        highlight_path: bool = False,
        path_color: str = "cyan",
        path_width: float = 2.5,
        path_node_size: float = 45.0,
        path_start_color: str = "lime",
        path_end_color: str = "red",
        path_node_outline: str = "black",
        node_show: bool = True,
        edge_show: bool = True,
        save_out: bool = True,
    ) -> None:
        """폴리곤/그래프/문-방 연결 라벨을 그리고 필요 시 이미지로 저장한다."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)
        ax_main: plt.Axes = axes[0]
        ax_lines: plt.Axes = axes[1]
        ax_lines.set_title("Graph Edge Distribution")
        ax_main.set_title(title)
        colors = {
            "room": "lightgray",
            "wall": "darkgray",
            "door": "saddlebrown",
            "stairs": "skyblue",
            "elevator": "indigo",
        }
        for room in self.objs.get("rooms", []):
            ax_main.add_patch(
                MplPolygon(room["corners"], facecolor=colors["room"], edgecolor="black", alpha=0.7)
            )
        for wall in self.objs.get("walls", []):
            ax_main.add_patch(MplPolygon(wall["corners"], facecolor=colors["wall"], edgecolor="black"))
        for door in self.objs.get("doors", []):
            ax_main.add_patch(MplPolygon(door["corners"], facecolor=colors["door"], edgecolor="black"))
        for stair in self.objs.get("stairs", []):
            ax_main.add_patch(MplPolygon(stair["corners"], facecolor=colors["stairs"], edgecolor="black"))
        for elevator in self.objs.get("elevators", []):
            ax_main.add_patch(
                MplPolygon(elevator["corners"], facecolor=colors["elevator"], edgecolor="black")
            )

        corridor_nodes_with_doors = {
            node_id
            for node_id, attrs in self.G.nodes(data=True)
            if attrs.get("type") == "door_endpoints" and attrs.get("door_link_ids")
        }

        if annotate_connections:
            for room in self.objs.get("rooms", []):
                label = f"R{room['id']}"
                doors = room.get("connected_door_ids")
                if doors:
                    label += f"\\nD{','.join(str(d_id) for d_id in doors)}"
                centroid = room.get("centroid")
                if centroid:
                    ax_main.text(
                        centroid.x,
                        centroid.y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=3,
                        color="black",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
                    )

            for door in self.objs.get("doors", []):
                label = f"D{door['id']}"
                rooms = door.get("connected_room_ids")
                if rooms:
                    label += f"\\nR{','.join(str(r_id) for r_id in rooms)}"
                centroid = door.get("centroid")
                if centroid:
                    ax_main.text(
                        centroid.x,
                        centroid.y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=4,
                        color="saddlebrown",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7),
                    )

            for node_id in corridor_nodes_with_doors:
                attrs = self.G.nodes[node_id]
                node_pos = attrs.get("pos")
                door_ids = attrs.get("door_link_ids", [])
                room_ids = attrs.get("room_link_ids", [])
                if not node_pos or not door_ids:
                    continue
                x, y = node_pos
                label = f"D{','.join(str(d_id) for d_id in door_ids)}"
                if room_ids:
                    label += f"\\nR{','.join(str(r_id) for r_id in room_ids)}"
                ax_main.text(
                    x,
                    y - 6,
                    label,
                    ha="center",
                    va="top",
                    fontsize=3,
                    color="deeppink",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.65),
                )

        pos = nx.get_node_attributes(self.G, "pos")
        node_colors: List[str] = []
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get("type")
            if node_type == "corridor":
                node_colors.append("green")
            elif node_type == "door_endpoints":
                node_colors.append("deeppink")
            else:
                node_colors.append("red")

        if node_show:
            nodes_coll = nx.draw_networkx_nodes(self.G, pos, node_size=0.1, node_color=node_colors, ax=ax_main)
            if hasattr(nodes_coll, "set_zorder"):
                nodes_coll.set_zorder(5)
        if edge_show:
            edges_coll = nx.draw_networkx_edges(self.G, pos, alpha=0.6, edge_color="orange", ax=ax_main)
            if isinstance(edges_coll, list):
                for coll in edges_coll:
                    if hasattr(coll, "set_zorder"):
                        coll.set_zorder(4)
            elif hasattr(edges_coll, "set_zorder"):
                edges_coll.set_zorder(4)

        if highlight_path and path:
            valid_path = [node for node in path if node in pos]
            if valid_path:
                path_nodes = nx.draw_networkx_nodes(
                    self.G,
                    pos,
                    nodelist=valid_path,
                    node_color=path_color,
                    node_size=path_node_size,
                    ax=ax_main,
                )
                if hasattr(path_nodes, "set_zorder"):
                    path_nodes.set_zorder(6)

                if len(valid_path) >= 2:
                    path_edges = list(zip(valid_path, valid_path[1:]))
                    path_edges_coll = nx.draw_networkx_edges(
                        self.G,
                        pos,
                        edgelist=path_edges,
                        edge_color=path_color,
                        width=path_width,
                        ax=ax_main,
                    )
                    if isinstance(path_edges_coll, list):
                        for coll in path_edges_coll:
                            if hasattr(coll, "set_zorder"):
                                coll.set_zorder(5)
                    elif hasattr(path_edges_coll, "set_zorder"):
                        path_edges_coll.set_zorder(5)

                start_node = valid_path[0]
                end_node = valid_path[-1]
                start_pos = pos[start_node]
                end_pos = pos[end_node]
                ax_main.scatter(
                    [start_pos[0]],
                    [start_pos[1]],
                    s=path_node_size * 1.5,
                    c=path_start_color,
                    edgecolors=path_node_outline,
                    linewidths=1.0,
                    zorder=7,
                )
                ax_main.scatter(
                    [end_pos[0]],
                    [end_pos[1]],
                    s=path_node_size * 1.5,
                    c=path_end_color,
                    edgecolors=path_node_outline,
                    linewidths=1.0,
                    zorder=7,
                )

        ax_main.set_aspect("equal", adjustable="box")
        ax_main.invert_yaxis()

        for room in self.objs.get("rooms", []):
            ax_lines.add_patch(
                MplPolygon(room["corners"], facecolor="none", edgecolor="silver", linewidth=0.6, linestyle="--")
            )
        for stair in self.objs.get("stairs", []):
            ax_lines.add_patch(
                MplPolygon(stair["corners"], facecolor="none", edgecolor="skyblue", linewidth=0.6, linestyle="--")
            )
        for elevator in self.objs.get("elevators", []):
            ax_lines.add_patch(
                MplPolygon(elevator["corners"], facecolor="none", edgecolor="indigo", linewidth=0.6, linestyle="--")
            )

        wall_segments = self.objs.get("wall_segments") or []
        if not wall_segments:
            for wall in self.objs.get("walls", []):
                corners = wall.get("corners") or []
                if len(corners) < 2:
                    continue
                for idx in range(len(corners)):
                    start = corners[idx]
                    end = corners[(idx + 1) % len(corners)]
                    if start == end:
                        continue
                    ax_lines.plot([start[0], end[0]], [start[1], end[1]], color="dimgray", linewidth=1.0, alpha=0.9)
        else:
            for seg in wall_segments:
                start = seg.get("start")
                end = seg.get("end")
                if not start or not end:
                    continue
                category = (seg.get("category") or "").lower()
                color = "black" if category in {"annotation", "line", "wall"} else "royalblue"
                linewidth = 1.2 if category in {"annotation", "line", "wall"} else 0.9
                ax_lines.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth, alpha=0.95)

        for door in self.objs.get("doors", []):
            corners = door.get("corners")
            if not corners:
                continue
            ax_lines.add_patch(MplPolygon(corners, fill=False, edgecolor=colors["door"], linewidth=1.0, linestyle="-"))

        ax_lines.set_aspect("equal", adjustable="box")
        ax_lines.invert_yaxis()
        ax_lines.set_xlabel("x")
        ax_lines.set_ylabel("y")
        ax_lines.grid(False)

        if output_path is not None and save_out:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300)
            print(f"그래프가 {output_path} 에 저장되었습니다.")

        if show_window:
            plt.show()
        else:
            plt.close(fig)
