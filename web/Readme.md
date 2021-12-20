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


## Google Cloud Platform(GCP)을 사용하실 경우
### 필수조건
- MySql
- GCP Cloud Storage
  - GCP key file 필요
- Secrets: ``` .streamlit/secrets.toml ``` 파일 필요
  - secrets.toml 파일은 개인정보 및 접근 권한에 대한 정보가 있으니 공유하시면 안됩니다.
- [mysql]의 정보는 GCP MySql에 접속하여 확인하실 수 있으며, \
  [gcp]의 정보는 key 파일을 만들면서 생성된 json 파일에서 확인하실 수 있습니다.
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

### 로컬 환경 셋팅
  1. ```secrets.toml``` 파일을 ```.streamlit``` 폴더에 넣습니다. 
  2. requirements.txt 파일을 실행하여 필요 라이브러리들을 설치해 줍니다. 
  ```
  pip install -r requirements.txt 
  ```
  3. 데이터 베이스 초기 셋팅을 해줍니다. ( DB Table setting ) 
  ``` 
  python init_database.py 
  ```
  4. streamlit을 실행시켜 줍니다. 
  ``` 
  streamlit run streamlit-app.py 
  ```
