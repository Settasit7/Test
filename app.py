import streamlit as st

st.set_page_config(page_title = 'Stocker', page_icon = ':chart_with_upwards_trend:' , layout = 'wide')

hide_st_style = '''
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
'''

st.markdown(hide_st_style, unsafe_allow_html = True)

@st.experimental_memo(suppress_st_warning=True)
def function():

    st.title('Stocker')
    st.write('##')
    st.write('เว็บแอปทำนายราคาปิดหุ้นในอนาคตด้วยปัญญาประดิษฐ์ :space_invader:')

    st.write('---')

    st.header('เริ่มการทำนาย')
    st.write('##')

    column1, column2, column3 = st.columns(3)

    with column1:
        st.write('##')
        st.write('1. เลือกหุ้น')
        st.write('##')
        name = st.text_input('กรอกชื่อย่อหุ้น', 'PTT.BK')
        st.write('##')
        st.write('หุ้นไทย: กรอกชื่อย่อตามด้วย.BK')
        st.write('หุ้นต่างประเทศ: กรอกชื่อย่อปกติ')

    with column2:
        st.write('##')
        st.write('2. ดึงข้อมูล')
        st.write('##')
        date1 = st.text_input('ดึงข้อมูลมาตั้งแต่วันที่เท่าไหร่', '2022-01-01')
        st.write('##')
        date2 = st.text_input('ดึงข้อมูลมาจนถึงวันที่เท่าไหร่', '2022-06-30')
        st.write('##')
        st.write('หมายเหตุ:')
        st.write('วันที่เริ่มต้นการดึงข้อมูลต้องอยู่ก่อนวันที่สิ้นสุดการดึงข้อมูล')

    with column3:
        st.write('##')
        st.write('3. ทำนายอนาคต')
        st.write('##')
        date3 = st.text_input('ทำนายราคาปิดจนถึงวันที่เท่าไหร่', '2022-08-31')
        st.write('##')
        day = st.number_input('ทำนายโดยใช้ข้อมูลย้อนหลังทั้งหมดกี่วัน', min_value = 1, value = 60)
        st.write('##')
        percent = st.number_input('แบ่งข้อมูลที่ดึงมาใช้ในการเทรนกี่เปอร์เซ็นต์', min_value = 1, max_value = 100, value = 90)
        st.write('##')
        st.write('หมายเหตุ:')
        st.write('จำนวนข้อมูลที่จะใช้ทำนายต้องน้อยกว่าหรือเท่ากับจำนวนข้อมูลที่ดึงมาจากขั้นตอนก่อนหน้า')
        st.write('ปรับเปอร์เซ็นต์ข้อมูลที่แบ่งมาใช้ในการเทรนเป็น 100 หากต้องการทำนายอนาคตจากข้อมูลที่เป็นปัจจุบันมากที่สุด ในกรณีนี้จะไม่แสดงกราฟช่วงเทส และแสดงค่า rmse เป็น 0')

    st.write('---')

    with st.spinner('แปป...'):

        import math
        import numpy as np
        import pandas as pd
        import yfinance as yf
        import matplotlib.pyplot as plt
        from keras.models import Sequential
        from keras.layers import Dense, LSTM
        from sklearn.preprocessing import MinMaxScaler
        plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

        ratio = percent / 100

        input = day
        dense = math.ceil(((2 / 3) * input) + 1)
        output = 1

        optimizer = 'nadam'
        loss = 'mean_squared_error'

        df0 = yf.download(name, date1, date2)
        df1 = df0.filter(['Close'])
        le0 = len(df1)

        date_list = [i.strftime('%Y-%m-%d 00:00:00') for i in pd.date_range(date2, date3, freq = 'B')]
        df2 = pd.DataFrame(date_list)
        df2.rename(columns = {df2.columns[0]: 'Date'}, inplace = True)
        df2['Predictions'] = np.nan
        df2.set_index('Date', inplace = True)
        df2.index = pd.to_datetime(df2.index)

        df3 = pd.concat([df1, df2])
        df3 = df3.fillna(np.nan)
        df4 = df3.filter(['Close'])
        ra1 = df4.values
        le1 = math.ceil(le0 * ratio)
        df3.reset_index(inplace = True)

        scaler = MinMaxScaler(feature_range = (0,1))
        sc1 = scaler.fit_transform(ra1)

        tr0 = sc1[0:le1,:]
        tr1 = []
        tr2 = []
        for i in range(day, le1):
            tr1.append(tr0[i-day:i,0])
            tr2.append(tr0[i,0])

        tr1, tr2 = np.array(tr1), np.array(tr2)
        tr1 = np.reshape(tr1, (tr1.shape[0], tr1.shape[1], 1))

        model = Sequential()
        model.add(LSTM(input, return_sequences = True, input_shape = (tr1.shape[1], 1)))
        model.add(LSTM(input, return_sequences = False))
        model.add(Dense(dense))
        model.add(Dense(output))
        model.compile(optimizer, loss)
        model.fit(tr1, tr2, batch_size = 1, epochs = 3)

        te0 = sc1[le1 - day:,:]
        le2 = len(te0)
        le = len(sc1)
        te1 = []
        te1.append(sc1[le1 - day:le1,0])
        te1 = np.array(te1)
        te1 = np.reshape(te1, (te1.shape[0], te1.shape[1], 1))

        for i in range(le1 + 1, le):
            re = model.predict(te1)
            ta = scaler.inverse_transform(re)
            ta = ta[0].tolist()[0]
            df3.loc[i, 'Predictions'] = ta
            re = re.tolist().pop(0).pop(0)
            te1 = te1.tolist().pop(0)
            te3 = []
            for j in range(day):
                te3.append(te1.pop(0).pop(0))
            te3.pop(0)
            te3.append(re)
            te4 = []
            te4.append(te3[:])
            te1 = te4
            te1 = np.array(te1)
            te1 = np.reshape(te1, (te1.shape[0], te1.shape[1], 1))

        te2 = ra1[le1:le0,:]

        df5 = df3

        df3.set_index('Date',inplace = True)
        df3.index = pd.to_datetime(df3.index)

        df6 = df5.iloc[le1 + 1:,:]

        da1 = df3.filter(['Close'])
        da2 = df3.filter(['Predictions'])
        li1 = da1[:le1]
        li2 = da1[le1:le0]
        li3 = da2[le1:le0]
        li4 = da2[le0:]

        rmse1 = 0
        for i in range(le1 + 1, le0):
            pop1 = df5.filter(['Close']).values.tolist()[i].pop(0)
            pop2 = df5.filter(['Predictions']).values.tolist()[i].pop(0)
            rmse0 = (pop1 - pop2) ** 2
            rmse1 = rmse1 + rmse0
        if le0 - le1 - 1 == 0:
            rmse = 'undefined'
        else:
            rmse2 = rmse1 / (le0 - le1 - 1)
            rmse = rmse2 ** 0.5

        fig = plt.figure(figsize = (20,8))
        plt.title(name, fontsize = 18)
        plt.xlabel('Date', fontsize = 18)
        plt.ylabel('Close Price THB', fontsize = 18)
        plt.plot(li1['Close'], c = 'violet')
        plt.plot(li2['Close'], c = 'darkviolet')
        plt.plot(li3['Predictions'], c = 'green')
        plt.plot(li4['Predictions'], c = 'limegreen')
        plt.legend(['Real Data (Training)', 'Real Data (Testing)', 'Prediction (Testing)', 'Prediction (Future)'], loc = 'lower left')

    st.success('เสร็จละ')

    st.header('ผลการทำนาย')
    st.write('##')

    st.write(fig)

    st.write(df6)

    st.write('Test Root Mean Square Error: ')
    st.write(rmse)

    st.write('---')

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

    local_css("style/style.css")

    st.header('มาคุยกัน')
    st.write('##')

    contact_form = '''
    <form action = 'https://formsubmit.co/44cb3e677835b6d430f0e09576b4dece' method = 'POST'>
        <input type = 'text' name = 'name' placeholder = 'ชื่อ' required>
        <input type = 'email' name = 'email' placeholder = 'อีเมล' required>
        <textarea name = 'message' placeholder = 'ข้อความ' required></textarea>
        <button type="submit">ส่ง</button>
    </form>
    '''

    st.markdown(contact_form, unsafe_allow_html = True)

function()

st.experimental_memo.clear()