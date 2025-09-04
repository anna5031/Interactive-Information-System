# 프로젝트 이름

이 프로젝트는 파이썬 기반으로 작성되었습니다.  
아래의 방법을 따라 환경을 세팅하고 실행할 수 있습니다.

---

## 1. 가상환경 생성 (권장)

```bash
python -m venv venv
```

- Windows:  
  ```bash
  venv\Scripts\activate
  ```

- macOS / Linux:  
  ```bash
  source venv/bin/activate
  ```

---

## 2. 의존성 설치

아래 명령어를 실행하면 `requirements.txt`에 명시된 패키지가 모두 설치됩니다.

```bash
pip install -r requirements.txt
```

---

## 3. 실행 방법

예시 (main.py 실행):

```bash
python src/main.py
```

---

## 4. 참고

- `requirements.txt` 파일은 `pip freeze > requirements.txt` 명령어로 생성되었습니다.
- 새로운 패키지를 설치했을 경우, 다시 위 명령어를 실행해 `requirements.txt`를 업데이트 해주세요.
