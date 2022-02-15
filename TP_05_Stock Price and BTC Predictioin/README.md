# Stock and BTC Price Prediction

---
### Goal
LSTM으로 삼성전자 주식, 비트코인 가격 예측하기

### Dependency
1. Python 3
2. Numpy : 행렬연산
3. Keras :  딥러닝 모델 작성
4. pandas : csv 파일 로드
5. matplotlib : 데이터 시각화
6. yfinance : yahoo finance에서 주식 종목별 ohlcv 데이터 수집
7. pyupbit : 업비트에 상장된 코인의 ohlcv 데이터 수집

### Model
LSTM(Long Short Term Memory)  

### Dataset
1. [Yahoo Finance](https://finance.yahoo.com/)
2. [CoinMarketCap](https://coinmarketcap.com/)

### Run & Result
- 내일의 주가 예측  => 
    ```
    stock.ipynb
    ```
- 내일의 코인 시세 예측 =>
    ```
    crypto.ipynb
    ```

### Study
- LSTM(Long Short Term Memory)  
   이전 데이터를 가지고 이후의 데이터를 예측하는 인공지능 모델

### Level up
1. 데이터 수집 방식  
  yfinance와 pyupbit를 사용하여 주가데이터와 코인 시세 데이터를 쉽고 정확하게 불러올 수 있다.<br>  
   **삼성전자 주가데이터 수집**
   ```python
   import yfinance as yf
   import pandas_datareader.data as pdr

   yf.pdr_override()

   #Get the stock starting date
   start_date = '07-02-2017'

   #Get the stock ending date
   end_date = '07-02-2022'
   
   start = datetime.strptime(start_date, '%d-%m-%Y')
   end = datetime.strptime(end_date, '%d-%m-%Y')
   
   #Create a dataframe to store the adjusted close price of the stocks
   data = pd.DataFrame()
   
   #삼성전자 주가데이터 가져오기
   data = pdr.get_data_yahoo('005930.KS', data_source='yahoo', start=start, end=end)   
   ```
   
   **비트코인 시세 데이터 수집**
   ```python
    import pyupbit
   
    #업비트에 상장된 코인 리스트 확인(원화 마켓의 암호화폐 선별)
    print(pyupbit.get_tickers(fiat="KRW"))
   
    #코인 현재가 확인
    print(pyupbit.get_current_price(['KRW-BTC', 'KRW-ETH']))
   
    #BTC의 과거 데이터 가져오기
    ticker = 'KRW-BTC'
    interval ='day'    # 봉 길이: 하루단위로 설정
    to = '2022-02-15'  # 데이터의 마지막 시점
    count = 365*3        # 몇 개의 봉을 가져올지 (3년치니까, 365*3 으로 설정)

    pyupbit.get_ohlcv(ticker=ticker, interval=interval, to=to, count=count)
   ```

### Reference
1. [`빵형의 개발도상국`님의 유튜브 영상](https://www.youtube.com/watch?v=sG_WeGbZ9A4&t=35s)
2. [`kairess`님의 github](https://github.com/kairess/stock_crypto_price_prediction.git)
