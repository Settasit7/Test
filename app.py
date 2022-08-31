import streamlit as st

st.set_page_config(page_title = 'Stocker', page_icon = ':space_invader:' , layout = 'wide')

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
    st.write('เว็บแอปทำนายราคาปิดหุ้นในอนาคตด้วยปัญญาประดิษฐ์')
    st.write(':globe_with_meridians: [English](https://settasit7-stocker-english-app-4dkkna.streamlitapp.com)')

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
        st.write('หมายเหตุ:')
        st.write('โปรแกรมจะดึงข้อมูลมาตั้งแต่วันที่ในช่องด้านบนจนถึงเมื่อวาน')

    with column3:
        st.write('##')
        st.write('3. ทำนายอนาคต')
        st.write('##')
        date3 = st.text_input('ทำนายราคาปิดจนถึงวันที่เท่าไหร่', '2022-09-30')
        st.write('##')
        st.write('หมายเหตุ:')
        st.write('โปรแกรมจะทำนายราคาปิดตั้งแต่วันแรกที่ดึงข้อมูลจนถึงวันที่ในช่องด้านบน')

    st.write('---')

    with st.spinner('แปป...'):

        from datetime import date

        date2 = date.today().strftime('%Y-%m-%d')

        import yfinance as yf

        df0 = yf.download(name, date1, date2)
        df0.reset_index(inplace = True)

        df1 = df0[['Date', 'Close']]
        df1 = df1.rename(columns = {'Date': 'ds', 'Close': 'y'})

        from fbprophet import Prophet

        m = Prophet()
        m.fit(df1)

        import pandas as pd

        df2 = [i.strftime('%Y-%m-%d 00:00:00') for i in pd.date_range(date2, date3, freq = 'B')]
        df2 = pd.DataFrame(df2)
        df2.rename(columns = {df2.columns[0]: 'ds'}, inplace = True)

        df3 = pd.concat([df1, df2])

        df4 = m.predict(df3)

        df4 = df4.rename(columns = {'ds': 'Date', 'yhat': 'Predict^', 'yhat_lower': 'Predict^Lower', 'yhat_upper': 'Predict^Upper'})
        df4 = df4[['Date', 'Predict^', 'Predict^Lower', 'Predict^Upper']]
        df4.set_index('Date',inplace = True)
        df4.index = pd.to_datetime(df4.index)

        df5 = df4[len(df0):]

        df0 = df0[['Date', 'Open', 'High', 'Low', 'Close']]
        df0.set_index('Date',inplace = True)
        df0.index = pd.to_datetime(df0.index)

        from plotly import graph_objs as go
        from plotly.subplots import make_subplots

        fig = make_subplots(specs = [[{"secondary_y": True}]])
        fig.update_layout(autosize = True)
        fig.add_trace(go.Scatter(x = df4.index, y = df4['Predict^'], name = 'Predict^', marker_color = 'blue'))
        fig.add_trace(go.Scatter(x = df4.index, y = df4['Predict^Upper'], name = 'Predict^Upper', marker_color = 'lightblue', showlegend = False))
        fig.add_trace(go.Scatter(x = df4.index, y = df4['Predict^Lower'], name = 'Predict^Lower', marker_color = 'lightblue', showlegend = False))
        fig.add_trace(go.Candlestick(x = df0.index, open = df0['Open'], high = df0['High'], low = df0['Low'], close = df0['Close'], name = 'Real Data'))
        df0.loc[df0['Close'] - df0['Open'] >= 0, 'color'] = 'green'
        df0.loc[df0['Close'] - df0['Open'] <   0, 'color'] = 'red'
        fig.update_xaxes(rangebreaks = [dict(bounds = ['sat', 'mon'])])
        fig.update_layout(title={'text': name, 'x': 0.5})

        import math
        import numpy as np

        y1 = df0.values[: , 0]
        y2 = df4.values[: len(df0), 0]
        RMSE = math.sqrt(np.square(np.subtract(y1, y2)).mean())

    st.success('เสร็จละ')

    st.header('ผลการทำนาย')
    st.write('##')

    st.plotly_chart(fig, use_container_width = True)

    st.write(df5)

    st.write('Root Mean Square Error: ')
    st.write(RMSE)

    st.write('---')

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

    local_css("style/style.css")

    st.header('มาคุยกัน')
    st.write('##')

    col1, col2 = st.columns(2)

    with col1:

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