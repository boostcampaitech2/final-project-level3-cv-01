## streamlit을 이용한 웹 프로토타입 입니다.

실행 전 ffmpeg를 설치(리눅스)
```
apt-get install ffmpeg libsm6 libxext6  -y
```


windows의 경우 Microsoft C++ Build Tools 설치

https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/


Dependency 설치
```
pip install -r requirements.txt
```

앱 실행
```
streamlit run streamlit-app.py
```


## Google Cloud Platform을 사용하실 경우
### 필수조건
- MySql
- GCP Cloud Storage
  - GCP key file 필요
- Secrets: ``` .streamlit/secrets.toml ``` 파일 필요
  - secrets.toml 파일은 개인정보 및 접근 권한에 대한 정보가 있으니 공유하시면 안됩니다.
```
#DO NOT SHARE THIS INFORMATION!!!!
  [mysql]
  host = <YOUR_HOST>
  port = 3306
  database = <YOUR_DATABASE>
  user = <YOUR_USER>
  password = <YOUR_PASSWORD>

  [gcp]
  project_id = <YOUR_PROJECT_ID>
  private_key_id = <YOUR_PROJECT_KEY>
  private_key = <YOUR_PRIVATE_KEY>
  client_email = <YOUR_CLIENT_EMAIL>
  client_id = <YOUR_CLIENT_ID>
  bucket = <YOUR_BUCKET>
```
