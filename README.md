# AirCompressor_anomaly_detection_AIFACTORY
https://aifactory.space/competition/detail/2226

- Airflow를 이용하여 ETL 파이프라인 구성 후, WandB로 모델링 모니터링.
- 참가 팀명: 신길동토니스타크
- 대회 종료 후 업로드

<br>

**대회 주제**\
■ 범천(주)은 ESG 가치를 담아 산업용 공기압축기를 개발하는 대덕연구개발특구 소재 기업입니다.

- 제4회 연구개발특구 AI SPARK 챌린지는 산업기기 피로도를 예측하는 문제입니다.
- 산업용 공기압축기 및 회전기기에서 모터 및 심부 온도, 진동, 노이즈 등은 기기 피로도에 영향을 주는 요소이며, 피로도 증가는 장비가 고장에 이르는 원인이 됩니다.
- 피로도 증가 시 데이터 학습을 통해 산업기기 이상 전조증상을 예측하여 기기 고장을 예방하고 그로 인한 사고를 예방하는 모델을 개발하는 것이 이번 대회의 목표입니다.

<br>

**모델 조건**
1. 본 대회의 모델링은 비지도학습 방식으로 진행됩니다.

2. 향후 실시간 판정에 활용될 수 있도록, 개발된 모델은 다음의 조건을 충족하여야 합니다.

- 입력된 데이터를 정상(0), 이상(1)로 구분하는 이진 분류 모델이어야 합니다.
- 시간 단위로 생성되는 입력 데이터에 대하여 판정을 수행할 수 있는 모델이어야 합니다.
- 신규 데이터로 학습/개선이 가능한 모델이어야 합니다.
- 총 8개의 대상 설비를 모델링하면서, 설비별로 별도의 모델을 학습하는 것은 허용되나 모두 동일한 아키텍처를 사용해야 합니다.
(예: 설비 1에 사용한 모델 구조를 나머지 설비에도 사용하여야 함)

<br>

**데이터 구성**
![image](https://user-images.githubusercontent.com/118624081/230829560-8a03ab71-a807-42e6-8def-8a3f6f827dd0.png)   
air_inflow: 공기 흡입 유량 (^3/min)   
air_end_temp: 공기 말단 온도 (°C)   
out_pressure: 토출 압력 (Mpa)   
motor_current: 모터 전류 (A)   
motor_rpm: 모터 회전수 (rpm)   
motor_temp: 모터 온도 (°C)   
motor_vibe: 모터 진동 (mm/s)   
type: 설비 번호   

설비 번호 [0, 4, 5, 6, 7]: 30HP(마력)   
설비 번호 1: 20HP   
설비 번호 2: 10HP   
설비 번호 3: 50HP   
