# STUBO 프로젝트 설치 및 실행 가이드

이 문서는 STUBO 프로젝트를 로컬 환경에서 실행하기 위한 단계별 가이드입니다.

---

## ✅ 사전 준비 사항

- Git 설치: [https://git-scm.com/](https://git-scm.com/)
- Docker Desktop 설치: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
- `.env` 파일은 개별 전달 예정입니다.

---

## 🛠️ 설치 및 실행 절차

### 1. 프로젝트 클론

GitHub에서 해당 브랜치(MSA)를 본인의 PC로 내려받습니다.

```bash
git clone -b MSA https://github.com/KKimHyeonJoo/STUBO.git
```

> ✅ 참고: 이미 clone 된 경우엔 git remote 방식으로 연결하지 않아도 됩니다.

---

### 2. Docker Desktop 설치

운영체제에 맞게 Docker Desktop을 설치해주세요.

- Windows: [다운로드 링크](https://www.docker.com/products/docker-desktop/)
- macOS: [다운로드 링크](https://www.docker.com/products/docker-desktop/)

설치 후 Docker 실행 상태 확인 필수!

---

### 3. `.env` 파일 위치시키기

전달받은 `.env` 파일을 `STUBO` 폴더의 **최상위 경로**에 위치시킵니다.

```
STUBO/
├── .env  👈 여기에!
├── main.py
├── gateway/
└── ...
```

> ❌ `gateway/`나 다른 하위 폴더에 두면 안 됩니다.

---

### 4. 커맨드 라인(cmd) 실행 → 프로젝트 폴더로 이동

예시:

```bash
cd C:\Users\user\MSA\STUBO
```

---

### 5. 모델 다운로드 및 폴더 생성

`save_model.py`를 실행하여 모델 파일을 자동 다운로드하고 `models/` 폴더를 생성합니다.

```bash
python -m save_model.py
```

> 실행 후 `STUBO/models/` 디렉토리가 생성되어야 합니다.

```
STUBO/
├── models/
│   ├── jhgan-ko-sroberta-multitask/
│   └── kosimcse-roberta-multitask/
```

---

### 6. Docker 이미지 빌드 (기본 requirements 설치)

```bash
docker build -t stubo-base:latest .
```

> `Dockerfile`을 기반으로 OCR, PDF 등 공통 패키지를 설치한 이미지가 생성됩니다.

---

### 7. Docker Compose로 서비스별 이미지 빌드

```bash
docker compose build
```

> `docker-compose.yaml`에 정의된 각 서비스의 requirements가 설치되며, 개별 Dockerfile이 실행됩니다.

---

### 8. Docker Compose로 전체 컨테이너 실행

```bash
docker compose up
```

> 여러 백엔드 서비스와 프론트엔드 서버가 동시에 실행됩니다.

---

### 9. 서비스 접속 테스트

브라우저에서 다음 주소로 접속해 메인 화면이 잘 나오는지 확인하세요:

```
http://localhost:8501
```

---

### 10. API 서버 상태 확인 (선택)

각 백엔드 API가 정상적으로 실행되고 있는지 개별 포트로 확인할 수 있습니다.

| 과목 | 포트 | 주소 |
|------|------|------|
| 문학 | 8001 | http://localhost:8001 |
| 비문학 | 8002 | http://localhost:8002 |
| 화법과 작문 | 8003 | http://localhost:8003 |
| 언어와 매체 | 8004 | http://localhost:8004 |

---

## 📌 기타 안내

- `.env` 파일은 GitHub에 포함되어 있지 않으며, 별도로 전달됩니다.
- `models` 폴더는 `save_model.py`를 통해 자동 다운로드 및 생성됩니다.
