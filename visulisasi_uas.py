import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualisasi():
    st.write("**Visualisasi Data**")

    pilihan = st.selectbox(
        "Pilih Jenis Data Visualisasi",
        ["Kesehatan", "Lingkungan"]
    )

    if pilihan == "Kesehatan":
        chart1()
    elif pilihan == "Lingkungan":
        chart2()

def chart1():
    df = pd.read_excel('data uas ML kesehatan.xlsx')

    # Rename biar enak dipakai
    df = df.rename(columns={
        'Y_Gallstone Status': 'gallstone',
        'X1_Age': 'age',
        'X2_Gender': 'gender',
        'X3_Comorbidity': 'comorbidity',
        'X4_Body Mass Index (BMI)': 'bmi',
        'X5_Visceral Fat Rating (VFR)': 'vfr',
        'X6_Glucose': 'glucose',
        'X7_Total Cholesterol (TC)': 'cholesterol',
        'X8_Vitamin D': 'vitamin_d'
    })

    # ================= METRIC =================
    total_pasien = df.shape[0]
    gallstone_count = df['gallstone'].sum()
    gallstone_rate = gallstone_count / total_pasien * 100

    col1, col2, col3, col4 = st.columns([2,2,3,1])
    with col1:
        st.metric("Total Pasien", total_pasien)
    with col2:
        st.metric("Pasien Gallstone", gallstone_count)
    with col3:
        st.metric("Persentase", f"{gallstone_rate:.2f}%")

    # ================= FILTER =================
    if 'selected_gender' not in st.session_state:
        st.session_state.selected_gender = None
    if 'selected_gallstone' not in st.session_state:
        st.session_state.selected_gallstone = None

    with col4:
        if st.button("🔄"):
            st.session_state.selected_gender = None
            st.session_state.selected_gallstone = None
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Gender**")
        if st.button("Laki-laki"):
            st.session_state.selected_gender = 1
        if st.button("Perempuan"):
            st.session_state.selected_gender = 0

    with col2:
        st.write("**Status Gallstone**")
        if st.button("Gallstone"):
            st.session_state.selected_gallstone = 1
        if st.button("Non-Gallstone"):
            st.session_state.selected_gallstone = 0

    # Apply filter
    filtered_df = df.copy()
    if st.session_state.selected_gender is not None:
        filtered_df = filtered_df[filtered_df['gender'] == st.session_state.selected_gender]
    if st.session_state.selected_gallstone is not None:
        filtered_df = filtered_df[filtered_df['gallstone'] == st.session_state.selected_gallstone]

    st.dataframe(filtered_df.head())

    # ================= BAR CHART =================
    st.write("**Distribusi Status Gallstone**")
    gallstone_bar = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('gallstone:N', title='Status Gallstone'),
        y=alt.Y('count():Q', title='Jumlah Pasien'),
        color=alt.Color('gallstone:N', legend=None),
        tooltip=['count():Q']
    ).properties(height=300)
    st.altair_chart(gallstone_bar, use_container_width=True)

    # ================= BOXPLOT =================
    col1, col2 = st.columns(2)

    with col1:
        st.write("**BMI vs Gallstone (Boxplot)**")
        box_bmi = alt.Chart(filtered_df).mark_boxplot().encode(
            x='gallstone:N',
            y='bmi:Q',
            color='gallstone:N'
        ).properties(height=300)
        st.altair_chart(box_bmi, use_container_width=True)

    with col2:
        st.write("**Cholesterol vs Gallstone (Boxplot)**")
        box_chol = alt.Chart(filtered_df).mark_boxplot().encode(
            x='gallstone:N',
            y='cholesterol:Q',
            color='gallstone:N'
        ).properties(height=300)
        st.altair_chart(box_chol, use_container_width=True)

    # ================= BARCHART =================
    st.write("**Distribusi Total Cholesterol (TC) berdasarkan Status Gallstone**")
    chol_hist = alt.Chart(filtered_df).mark_bar(opacity=0.7).encode(
        x=alt.X('cholesterol:Q', bin=alt.Bin(maxbins=30), title='Total Cholesterol (TC)'),
        y=alt.Y('count():Q', title='Jumlah Pasien'),
        color=alt.Color('gallstone:N', title='Gallstone'),
        tooltip=['gallstone:N', alt.Tooltip('count():Q', title='Jumlah')]
    ).properties(height=300)

    st.altair_chart(chol_hist, use_container_width=True)

    # ================= SCATTER =================
    st.write("**Hubungan BMI & Glukosa**")
    scatter = alt.Chart(filtered_df).mark_circle(size=70).encode(
        x=alt.X('glucose:Q', title='Glukosa'),
        y=alt.Y('bmi:Q', title='BMI'),
        color=alt.Color('gallstone:N', title='Gallstone'),
        tooltip=['age', 'bmi', 'glucose', 'cholesterol']
    ).interactive().properties(height=350)
    st.altair_chart(scatter, use_container_width=True)

    st.write("**Hubungan Glukosa & Total Cholesterol (TC)**")
    scatter_gc = alt.Chart(filtered_df).mark_circle(size=70).encode(
        x=alt.X('glucose:Q', title='Glukosa'),
        y=alt.Y('cholesterol:Q', title='Total Cholesterol (TC)'),
        color=alt.Color('gallstone:N', title='Gallstone'),
        tooltip=['age', 'glucose', 'cholesterol', 'bmi', 'gallstone']
    ).interactive().properties(height=350)
    st.altair_chart(scatter_gc, use_container_width=True)

    # ================= HEATMAP KORELASI =================
    st.write("**Heatmap Korelasi Variabel Kesehatan**")
    corr = filtered_df[['age','bmi','vfr','glucose','cholesterol','vitamin_d','gallstone']].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
def chart2():
    df = pd.read_excel('data uas ML Lingkungan.xlsx')

    # ================= METRIC =================
    avg_wri = df['WRI'].mean()
    max_wri = df['WRI'].max()
    min_wri = df['WRI'].min()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rata-rata WRI", f"{avg_wri:.2f}")
    col2.metric("WRI Tertinggi", f"{max_wri:.2f}")
    col3.metric("WRI Terendah", f"{min_wri:.2f}")

    # ================= FILTER =================
    region_list = sorted(df['Region'].unique().tolist())
    region_list.insert(0, 'ALL')   # 👉 tambah ALL

    selected_region = st.multiselect(
        "Pilih Region",
        region_list,
        default=['ALL']
    )

    # 👉 LOGIC FILTER ALL
    if 'ALL' in selected_region or len(selected_region) == 0:
        filtered_df = df.copy()
    else:
        filtered_df = df[df['Region'].isin(selected_region)]

    st.dataframe(filtered_df.head())

    # ================= BOXPLOT =================
    st.write("**Distribusi Indikator Risiko (Box Plot)**")
    box_cols = [
        'Exposure',
        'Vulnerability',
        'Susceptibility',
        'Lack of Coping Capabilities']
    box_df = filtered_df.melt(
        id_vars='Region',
        value_vars=box_cols,
        var_name='Indikator',
        value_name='Nilai')
    boxplot = alt.Chart(box_df).mark_boxplot().encode(
        x=alt.X('Indikator:N', title='Indikator Risiko'),
        y=alt.Y('Nilai:Q', title='Nilai'),
        color=alt.Color('Indikator:N', legend=None),
        tooltip=['Indikator', 'Nilai']).properties(height=400)
    st.altair_chart(boxplot, use_container_width=True)

    # ================= BAR CHART =================
    st.write("**Top 10 Negara dengan WRI Tertinggi**")

    top10 = df.sort_values('WRI', ascending=False).head(10)

    bar_wri = alt.Chart(top10).mark_bar().encode(
        x=alt.X('WRI:Q', title='World Risk Index'),
        y=alt.Y('Region:N', sort='-x', title='Region'),
        color=alt.Color('WRI:Q', scale=alt.Scale(scheme='reds')),
        tooltip=['Region', 'WRI']
    ).properties(height=350)

    st.altair_chart(bar_wri, use_container_width=True)

    # ================= SCATTER / BUBBLE =================
    st.write("**Exposure vs Vulnerability (Bubble = WRI)**")

    bubble = alt.Chart(filtered_df).mark_circle().encode(
        x=alt.X('Exposure:Q', title='Exposure'),
        y=alt.Y('Vulnerability:Q', title='Vulnerability'),
        size=alt.Size('WRI:Q', scale=alt.Scale(range=[100, 1500])),
        color=alt.Color('WRI:Q', scale=alt.Scale(scheme='oranges')),
        tooltip=['Region','Exposure','Vulnerability','WRI']
    ).interactive().properties(height=400)

    st.altair_chart(bubble, use_container_width=True)

    # ================= RADAR CHART =================
    st.write("**Profil Risiko Region (Radar Chart)**")

    radar_region = st.selectbox(
        "Pilih Region untuk Radar Chart",
        df['Region'].unique()
    )

    radar_cols = [
        'Exposure',
        'Susceptibility',
        'Lack of Coping Capabilities',
        'Lack of Adaptive Capacities'
    ]

    radar_data = df[df['Region'] == radar_region][radar_cols].values.flatten()
    radar_data = np.append(radar_data, radar_data[0])

    angles = np.linspace(0, 2*np.pi, len(radar_cols), endpoint=False)
    angles = np.append(angles, angles[0])

    fig, ax = plt.subplots(subplot_kw={'polar': True})
    ax.plot(angles, radar_data, linewidth=2)
    ax.fill(angles, radar_data, alpha=0.3)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, radar_cols)
    ax.set_title(f"Profil Risiko: {radar_region}")

    st.pyplot(fig)


