import subprocess
import sys
from pathlib import Path
from typing import Dict


class PathNavigationSystem:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.map_script_path = self.base_dir / "generate_map.py"
        self.maps_dir = self.base_dir / "maps"
        self.maps_dir.mkdir(exist_ok=True)
        self._create_map_generator()

    def _create_map_generator(self) -> None:
        """지도 생성 스크립트 생성"""
        map_script_content = '''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

def generate_building_map(destination, output_path="building_map.png"):
    """기계공학과 건물 지도 생성"""

    building_layout = {
        "기계관 101호": (1, 1, "1층"),
        "기계관 203호": (2, 3, "2층"),
        "기계관 205호": (2, 5, "2층"),
        "기계관 301호": (3, 1, "3층"),
        "기계관 305호": (3, 5, "3층"),
        "기계관 412호": (4, 2, "4층"),
        "창의세미나실 A": (1, 8, "1층"),
        "대회의실": (2, 8, "2층"),
        "토론실 B": (3, 8, "3층"),
        "기계공학과 사무실": (1, 5, "1층")
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    floor_colors = {
        "1층": "#FFE6E6",
        "2층": "#E6F3FF",
        "3층": "#E6FFE6",
        "4층": "#FFFFE6"
    }

    for room, (floor, x, floor_name) in building_layout.items():
        color = floor_colors.get(floor_name, "#F0F0F0")

        if destination in room:
            color = "#FF6B6B"
            linewidth = 3
        else:
            linewidth = 1

        rect = patches.Rectangle((x-0.4, floor-0.4), 0.8, 0.8,
                               linewidth=linewidth, edgecolor='black',
                               facecolor=color)
        ax.add_patch(rect)

        ax.text(
            x,
            floor,
            room.split()[-1],
            ha="center",
            va="center",
            fontsize=8,
            weight="bold" if destination in room else "normal",
        )

    for floor_name, color in floor_colors.items():
        floor_num = int(floor_name[0])
        ax.text(
            0,
            floor_num,
            floor_name,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color),
        )

    if destination in building_layout:
        dest_floor, dest_x, dest_floor_name = building_layout[destination]

        ax.scatter(5, 1, c="green", s=100, marker="o", label="입구")
        ax.text(5, 0.7, "입구", ha="center", va="top", fontsize=10, weight="bold")

        if dest_floor_name != "1층":
            ax.annotate(
                "",
                xy=(9, dest_floor),
                xytext=(9, 1),
                arrowprops=dict(arrowstyle="->", lw=2, color="red"),
            )
            ax.text(
                9.2,
                (dest_floor + 1) / 2,
                f"{dest_floor_name}\\n계단",
                ha="left",
                va="center",
                fontsize=9,
            )

            ax.annotate(
                "",
                xy=(dest_x, dest_floor),
                xytext=(9, dest_floor),
                arrowprops=dict(arrowstyle="->", lw=2, color="blue"),
            )
        else:
            ax.annotate(
                "",
                xy=(dest_x, dest_floor),
                xytext=(5, 1),
                arrowprops=dict(arrowstyle="->", lw=2, color="blue"),
            )

        ax.scatter(dest_x, dest_floor, c="red", s=150, marker="*", label="목적지")

    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_title(f"기계공학과 건물 - {destination} 찾아가기", fontsize=16, weight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        destination = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "building_map.png"
        generate_building_map(destination, output_path)
        print(f"Map generated: {output_path}")
    else:
        print("Usage: python generate_map.py <destination> [output_path]")
'''

        with self.map_script_path.open("w", encoding="utf-8") as f:
            f.write(map_script_content)

    def generate_navigation_map(self, destination: str) -> Dict:
        """경로 안내 지도 생성"""
        try:
            safe_name = destination.replace(" ", "_")
            output_path = self.maps_dir / f"map_{safe_name}.png"
            result = subprocess.run(
                [sys.executable, str(self.map_script_path), destination, str(output_path)],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "map_path": str(output_path),
                    "message": f"{destination}까지의 경로 지도가 생성되었습니다.",
                }

            return {"success": False, "error": result.stderr, "message": "지도 생성 중 오류가 발생했습니다."}

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "message": "지도 생성 시스템에 문제가 발생했습니다.",
            }

    def should_offer_navigation(self, intent: str, query: str, response: str) -> bool:
        request_keywords = ["경로", "안내", "길찾", "어디", "가는 법", "위치", "찾아가"]
        direct_request = any(keyword in query for keyword in request_keywords)

        location_keywords = ["호", "층", "위치", "연구실", "강의실", "세미나실"]
        has_location = any(keyword in response for keyword in location_keywords)

        eligible_intent = intent in ["professor_info", "class_info", "seminar_recommendation"]
        return eligible_intent and (direct_request or has_location)
