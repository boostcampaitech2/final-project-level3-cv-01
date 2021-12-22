# CCTV app for Android
안드로이드 디바이스에서 영상을 서버로 전송하기 위한 앱입니다.<br/>
이 앱은 실시간으로 PM 위법 행위를 탐지하기 위한 CCTV의 용도로 사용될 수 있습니다.<br/>
Socket 통신을 통해 서버에 촬영한 영상을 실시간으로 전송합니다.<br/>
해당 앱은 SDK 버전 Minimum 16, Target 31 기준으로 구현되었습니다.
<br/><br/><br/>

## Installation
cctv_app.apk를 이용해 앱을 설치하시면 됩니다.<br/>
SDK 버전이나 Android 버전이 최신일 경우, 카메라 보안의 문제로 앱 사용이 불가능할 수 있습니다.
<br/><br/><br/>

## Usage
#### Permissions
앱 사용을 위해 카메라 권한 사용과 저장소 접근 권한이 필요합니다.
<br/>

#### Connect
접속하고자 하는 서버의 IP 주소와 Port 번호를 입력하고 Connect 버튼을 누릅니다.<br/>
서버에서는 server/server.py가 실행되고 있어야 서버에 연결할 수 있습니다.<br/>
<br/>
<img src="https://user-images.githubusercontent.com/49871110/147073788-5ff5a62c-fa81-43ca-9a21-45dc4e8b80a7.jpg" width="200" height="400"/>
<br/><br/>

#### Recording
서버에 연결되었다면 새로운 Activity가 시작됩니다.<br/>
<br/>
<img src="https://user-images.githubusercontent.com/49871110/147074456-b7f3b7dd-ec15-4c1f-82bf-3061336ca046.jpg" width="200" height="400"/>
<br/><br/>

#### SEND
START 버튼을 누르면 실시간 영상 전송이 시작되고, 버튼이 STOP으로 바뀝니다.<br/>
STOP 버튼을 누르면 영상 전송을 중지할 수 있습니다.<br/>
<p float="left">
  <img src="https://user-images.githubusercontent.com/49871110/147074620-95d353e8-9f59-430f-92ee-822854314906.jpg" width="200" height="400"/>
  <img src="https://user-images.githubusercontent.com/49871110/147074612-fa86f9f5-5797-4c0f-8ec1-02e90712d820.jpg" width="200" height="400"/>
</p>
<br/>

#### FOCUS
FOCUS 버튼을 누르면 카메라 초점을 맞출 수 있습니다.
<p float="left">
  <img src="https://user-images.githubusercontent.com/49871110/147073928-5cf80a83-406e-49f8-bd21-818eb145f262.jpg" width="200" height="400"/>
  <img src="https://user-images.githubusercontent.com/49871110/147073957-45936de6-b292-4cae-b314-fb54f1c4b970.jpg" width="200" height="400"/>
</p>
<br/>

#### QUIT
QUIT 버튼을 누르면 영상 전송을 중지하고 Activity를 종료합니다.<br/>
Activity가 종료되면서 서버와의 연결도 해제됩니다.

<img src="https://user-images.githubusercontent.com/49871110/147075090-22701ab1-8bda-4cc3-ba3f-729a13f10738.png" width="200" height="400"/>
