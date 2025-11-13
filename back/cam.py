import cv2
import subprocess
import re


def get_avfoundation_camera_names():
    """
    ffmpeg -f avfoundation -list_devices true -i "" 를 호출해서
    macOS AVFoundation 비디오 디바이스 이름들을 가져온다.

    리턴: {인덱스(int): 이름(str)} 딕셔너리
    """
    try:
        # ffmpeg는 장치 목록을 stderr에 출력한다
        proc = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            "ffmpeg가 설치되어 있지 않습니다. `brew install ffmpeg` 후 다시 시도하세요."
        )
        return {}

    # 일반적으로 stderr에 장치 목록이 들어 있음
    text = proc.stderr + proc.stdout

    names = {}
    in_video_section = False

    for line in text.splitlines():
        line = line.strip()

        # 비디오 장치 섹션 시작
        if "AVFoundation video devices:" in line:
            in_video_section = True
            continue

        # 오디오 장치 섹션 오면 비디오 파싱 종료
        if "AVFoundation audio devices:" in line:
            break

        if in_video_section:
            # 예: [0] FaceTime HD Camera
            m = re.search(r"\[(\d+)\]\s+(.+)$", line)
            if m:
                idx = int(m.group(1))
                name = m.group(2).strip()
                names[idx] = name

    return names


def list_cameras_with_names():
    # ffmpeg로 얻은 카메라 인덱스-이름 매핑
    cam_names = get_avfoundation_camera_names()

    if not cam_names:
        print("카메라 이름 정보를 가져오지 못했습니다. 인덱스만 출력합니다.")
        cam_names = {}

    # ffmpeg에서 얻은 최대 인덱스를 기준으로 순회
    max_index = max(cam_names.keys()) + 1 if cam_names else 10

    available = []

    for i in range(max_index):
        # 맥에서는 avfoundation 백엔드 지정하는 게 안정적
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)

        if cap.isOpened():
            name = cam_names.get(i, "이름 알 수 없음")
            print(f"Camera index {i} 사용 가능 - {name}")
            available.append(i)
            cap.release()
        else:
            print(f"Camera index {i} 사용 불가")

    print("사용 가능한 카메라 인덱스:", available)
    return available


if __name__ == "__main__":
    list_cameras_with_names()
