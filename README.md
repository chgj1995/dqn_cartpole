# 가상환경(venv) 설정 및 의존성 설치

다음은 Windows 환경에서 Python 3.10을 사용해 `venv`라는 이름의 가상환경을 생성하고, 활성화한 뒤 `requirements.txt`에 명시된 패키지를 설치하는 과정입니다.

## 1. 가상환경 생성
```bash
py -3.10 -m venv venv
```

## 2. 가상환경 활성화
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# 또는 커맨드 프롬프트
venv\Scripts\activate
```

## 3. 의존성 설치
```bash
pip install -r requirements.txt
```