import streamlit as st
import pandas as pd
import numpy as np

st.header('Machine Learning Classification Penyakit Batu Empedu and Clustering Negara Berdasarkan Resiko Bencana')
st.write('**Dosen Pengampu:** Bapak Saeful Amri., S.Kom., M.Kom ')
st.write('**Nama :** Febrian Hikmah Nur Rohim')
st.write('**Nim :** B2D023016')
st.write('**S1 Sains Data** - Universitas Muhammadiyah Semarang')
st.write('**Semarang, 26 Desember 2025**')

tab1, tab2, tab3, tab4, tab5 ,tab6 = st.tabs([
                            'About Dataset', 
                            'Dashboard', 
                            'Machine Learning',
                            'Cara Kerja Model Terbaik',
                            'Prediction App',
                            'Contact Me'])

with tab1:
    import tentang_data
    tentang_data.about_dataset()

with tab2:
    import visulisasi_uas
    visulisasi_uas.visualisasi()

with tab3:
    import ML_uas
    ML_uas.ML()

with tab4:
    import tahapan_model
    tahapan_model.Tahapan_model()

with tab5:
    import prediksi
    prediksi.Prediksi_final()

with tab6:
    import tentang_gua
    tentang_gua.contact_me()
    
