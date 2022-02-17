# Audio Profanity Filter
Automatically bleep out the vulgar languages using Google Cloud STT and Python

---
### Goal
Speech-To-Text(STT)를 이용하여 욕을 하면 자동으로 삐- 처리하는 프로그램만들기

### Dependency
1. Python 3
2. google-cloud-speech
3. numpy
4. pydub : 오디오 편집 패키지

### Model
STT(Speech-To-Text)  
- Google Cloud Platform의 Cloud Speech API

### Data
욕설이 담긴 mp3 음원파일

### Run & Result
'fuck'이 나올 때 자동으로 삐- 처리
```python
main.ipynb
```

### Study
1. STT(Speech-To-Text)  
   - Google AI 기술로 지원되는 API를 사용하여 음성을 텍스트로 변환하는 기술
   - Google Cloud Platform(GCP)의 Cloud Speech API를 이용
   - 기존의 Google Assistant API와는 다름
   - Key 발급을 통해 쉽게 사용할 수 있는 Cloud API
   - 적당히 조용한 환경에서의 인식률은 DeepSpeech와 겨루지만, 노이즈 환경에서는 부족함<br><br>
2. Google Cloud STT 사용법  
    [STT 빠르게 시작하기](https://cloud.google.com/speech-to-text/docs/transcribe-client-libraries)  
    [STT docs(짧은 오디오 파일을 텍스트로 변환)](https://cloud.google.com/speech-to-text/docs/sync-recognize) <br><br> 

3. [pydub](https://github.com/jiaaro/pydub)
   - [docs](https://github.com/jiaaro/pydub/blob/master/API.markdown)
   - 쉽게 오디오를 조작할 수 있는 API
   - `AudioSegment.overlay()` : 오디오 위에 다른 오디오를 입히는 기능
     ```python
     sound.overlay(beep, position = swear_timeline[i][0], gain_during_overlay = -20)
     ```
      - sound: 오리지널 오디오  
      - beep: 오리지널 오디오 위에 올라갈 오디오  
      - position: beep이 올라가는 위치  
      - gain_during_overlay: 오리지널 오디오의 볼륨을 얼만큼 줄일것인지<br><br>
   - `AudioSegment.export()`: AudioSegment를 mp3 파일로 내보내기 
       ```python
        mixed_final.export('result/result_fxck.mp3', format='mp3')
       ```
   
4. [DeepSpeech](https://github.com/mozilla/DeepSpeech.git)  
   - [docs](https://deepspeech.readthedocs.io/en/r0.9/?badge=latest)  
   - [Baidu의 Deep Speech 연구 논문](https://arxiv.org/abs/1412.5567) 기반의 머신러닝 기법에 의해 훈련된 모델을 사용하는 오픈 소스 STT 엔진
    

### Level Up
1. 노이즈의 영향  
    한국어 음원파일로 테스트했을때 STT 성능이 좋지 않아서 영어 사운드로 테스트를 해 보았으나,  
    마찬가지로 많은 단어가 text로 변환되지 않았고, 변환된 내용 또한 정확도가 떨어지는 것을 확인할 수 있었다.  
    찾아보니, 오디오에 노이즈가 있거나 노이즈를 없애기 위한 전처리를 하는 경우에 정확도가 떨어지기 때문에  
    Google Cloud의 STT는 왜곡이나 노이즈 없이 선명한 오디오를 사용하는 것을 권장하고 있다.<br><br>

2. `AudioSegment.from_file('파일명', format='mp3')`로 파일 불러오려면 FileNotFoundError 발생  
-> 해결방법
    - ffmpeg.exe, ffprobe.exe,  ffplay.exe 설치 후 동일위치에 저장
    - `AudioSegment.from_file('파일명', format='mp3')` 대신에 `AudioSegment.from_mp3('파일명')` 으로 작성
      ```python
      AudioSegment.converter = r'C:\Users\dorot\PycharmProjects\KaggleStudy\TP_07_Audio Profanity Filter\ffmpeg.exe'                        
      AudioSegment.ffprobe   = r'C:\Users\dorot\PycharmProjects\KaggleStudy\TP_07_Audio Profanity Filter\ffprobe.exe'
      
      sound = AudioSegment.from_mp3('data/slang_eng.mp3')
      ```
    
### Reference
1. [`빵형의 개발도상국`님의 유튜브 영상](https://www.youtube.com/watch?v=J01pGSPOQTk&list=PL-xmlFOn6TUJ9KjFo0VsM3BI9yrCxTnAz&t=3s)
2. [`kairess`님의 github](https://github.com/kairess/audio-profanity-filter)