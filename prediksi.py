import streamlit as st

def Prediksi_final():
    st.write("**Aplikasi Prediksi Lachine Learning**")
    
    pilihan = st.selectbox(
        "Pilih Jenis Prediksi Lachine Learning",
        ["klasifikasi", "Klastering"]
    )

    if pilihan == "klasifikasi":
        app_prediksi_batu_empedu()
    elif pilihan == "Klastering":
        app_prediksi_Klasifikasi_wilayah()

def app_prediksi_batu_empedu():
    import streamlit as st
    import pandas as pd
    import pickle

    # st.set_page_config(page_title="Prediksi Batu Empedu", layout="wide")
    st.title("🩺 Aplikasi Prediksi Risiko Batu Empedu")
    st.caption("Input numerik + interpretasi kategori klinis")

    # =========================
    # LOAD MODEL
    # =========================
    with open("logistic_regression_gallstone.pkl", "rb") as f:
        model_bundle = pickle.load(f)

    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    features = model_bundle["features"]

    # =========================
    # FUNGSI PREDIKSI
    # =========================
    def prediksi_klasifikasi(input_df):
        X_scaled = scaler.transform(input_df)
        prob = model.predict_proba(X_scaled)[0, 1]
        pred = model.predict(X_scaled)[0]
        return pred, prob

    # =========================
    # FUNGSI KATEGORISASI
    # =========================
    def kategori_age(x):
        return "Dewasa Muda" if x < 40 else "Paruh Baya" if x < 60 else "Lansia"

    def kategori_bmi(x):
        if x < 18.5:
            return "Underweight"
        elif x < 25:
            return "Normal"
        elif x < 30:
            return "Overweight"
        else:
            return "Obese"

    def kategori_vfr(x):
        return "Normal" if x < 10 else "Tinggi" if x < 15 else "Sangat Tinggi"

    def kategori_glucose(x):
        return "Normal" if x < 100 else "Prediabetes" if x < 126 else "Diabetes"

    def kategori_chol(x):
        return "Normal" if x < 200 else "Borderline" if x < 240 else "High"

    def kategori_vitd(x):
        return "Deficiency" if x < 20 else "Insufficiency" if x < 30 else "Sufficient"

    def kategori_comorb(x):
        return "Tidak Ada" if x == 0 else "Ringan" if x == 1 else "Sedang" if x <= 3 else "Berat"

    # =========================
    # INPUT USER
    # =========================
    st.subheader("📋 Input Data Pasien")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Usia (Tahun)", 18, 100, 45)
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

    with col2:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        glucose = st.number_input("Glukosa Darah", 50.0, 300.0, 90.0)

    with col3:
        comorbidity = st.number_input("Jumlah Comorbidity", 0, 5, 1)
        vfr = st.number_input("Visceral Fat Rating (VFR)", 1.0, 30.0, 10.0)

    chol = st.number_input("Total Cholesterol", 100.0, 400.0, 180.0)
    vit_d = st.number_input("Vitamin D", 5.0, 100.0, 30.0)

    gender_val = 1 if gender == "Male" else 0

    # =========================
    # DATAFRAME MODEL
    # =========================
    input_data = pd.DataFrame([[
        age,
        gender_val,
        comorbidity,
        bmi,
        vfr,
        glucose,
        chol,
        vit_d
    ]], columns=features)

    # =========================
    # PREDIKSI
    # =========================
    if st.button("🔍 Prediksi Risiko Batu Empedu"):
        pred, prob = prediksi_klasifikasi(input_data)

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        # Kategori risiko
        if prob < 0.30:
            risk = "🟢 Risiko Rendah"
        elif prob < 0.60:
            risk = "🟡 Risiko Sedang"
        else:
            risk = "🔴 Risiko Tinggi"

        colA, colB = st.columns(2)
        colA.metric("Probabilitas Batu Empedu", f"{prob:.2%}")
        colB.metric("Kategori Risiko", risk)

        # =========================
        # TAMPILKAN KATEGORI KLINIS
        # =========================
        st.subheader("🧾 Interpretasi Kategori Klinis")

        kategori_df = pd.DataFrame({
            "Variabel": [
                "Usia", "BMI", "VFR", "Glukosa",
                "Total Cholesterol", "Vitamin D", "Comorbidity"
            ],
            "Nilai": [
                age, bmi, vfr, glucose, chol, vit_d, comorbidity
            ],
            "Kategori Klinis": [
                kategori_age(age),
                kategori_bmi(bmi),
                kategori_vfr(vfr),
                kategori_glucose(glucose),
                kategori_chol(chol),
                kategori_vitd(vit_d),
                kategori_comorb(comorbidity)
            ]
        })

        st.table(kategori_df)

        if pred == 1:
            st.error("⚠️ Pasien diprediksi berisiko mengalami batu empedu.")
        else:
            st.success("✅ Pasien diprediksi berisiko rendah mengalami batu empedu.")

        def rekomendasi_klinis(age, bmi, vfr, glucose, chol, vit_d, comorb):
            rekom = []

            # BMI
            if bmi < 18.5:
                rekom.append("Pasien tergolong kurus. Disarankan meningkatkan asupan nutrisi seimbang dan evaluasi status gizi.")
            elif bmi < 25:
                rekom.append("BMI dalam rentang normal. Pertahankan pola makan dan aktivitas fisik yang sehat.")
            elif bmi < 30:
                rekom.append("Pasien overweight. Disarankan pengaturan pola makan dan peningkatan aktivitas fisik.")
            else:
                rekom.append("Pasien obesitas. Dianjurkan program penurunan berat badan dan konsultasi gizi.")

            # VFR
            if vfr >= 15:
                rekom.append("Lemak viseral sangat tinggi. Perlu pengendalian berat badan dan aktivitas fisik rutin.")
            elif vfr >= 10:
                rekom.append("Lemak viseral tinggi. Disarankan diet rendah lemak dan olahraga teratur.")

            # Glukosa
            if glucose >= 126:
                rekom.append("Kadar glukosa tinggi (indikasi diabetes). Disarankan pemeriksaan lanjutan dan kontrol gula darah.")
            elif glucose >= 100:
                rekom.append("Kadar glukosa borderline. Disarankan modifikasi gaya hidup.")

            # Kolesterol
            if chol >= 240:
                rekom.append("Kolesterol total tinggi. Dianjurkan diet rendah lemak jenuh dan pemeriksaan lipid lanjutan.")
            elif chol >= 200:
                rekom.append("Kolesterol borderline. Perlu pemantauan dan pengaturan pola makan.")

            # Vitamin D
            if vit_d < 20:
                rekom.append("Defisiensi Vitamin D. Disarankan paparan sinar matahari dan/atau suplementasi sesuai anjuran medis.")
            elif vit_d < 30:
                rekom.append("Vitamin D belum optimal. Disarankan peningkatan paparan sinar matahari.")

            # Comorbidity
            if comorb >= 4:
                rekom.append("Komorbiditas tinggi. Disarankan pemantauan kesehatan secara rutin dan terintegrasi.")
            elif comorb >= 2:
                rekom.append("Terdapat beberapa penyakit penyerta. Perlu pengelolaan kondisi secara komprehensif.")

            return rekom

        st.subheader("💡 Rekomendasi Kesehatan")

        rekomendasi = rekomendasi_klinis(
            age, bmi, vfr, glucose, chol, vit_d, comorbidity)

        if rekomendasi:
            for r in rekomendasi:
                st.write(f"• {r}")
        else:
            st.write("Kondisi pasien secara umum berada dalam batas normal.")
        
        st.info("Rekomendasi bersifat edukatif dan tidak menggantikan diagnosis atau konsultasi dokter.")

# app_prediksi_batu_empedu()

def app_prediksi_Klasifikasi_wilayah():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import pickle

    st.title("🔍 Prediksi Pengelompokan Wilayah Kerentanan Bencana")

    # =========================
    # Load Model K-Means
    # =========================
    with open("kmeans_clustering_model.pkl", "rb") as f:
        model_bundle = pickle.load(f)

    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    features = model_bundle["features"]
    n_clusters = model_bundle["n_clusters"]

    st.subheader("📥 Input Indikator Wilayah")

    # =========================
    # Input User
    # =========================
    input_data = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        if i % 2 == 0:
            input_data[feature] = col1.number_input(
                feature, min_value=0.0, step=0.1
            )
        else:
            input_data[feature] = col2.number_input(
                feature, min_value=0.0, step=0.1
            )

    input_df = pd.DataFrame([input_data])

    # =========================
    # Prediksi
    # =========================
    if st.button("🔮 Prediksi Klaster"):
        X_scaled = scaler.transform(input_df)

        # ===== Prediksi K-Means =====
        cluster_result = model.predict(X_scaled)[0]

        # =========================
        # Output
        # =========================
        st.subheader("📊 Hasil Prediksi Klaster")

        st.metric(
            label=" ",
            value=f"Klaster {cluster_result}"
        )

        # =========================
        # Interpretasi Klaster
        # =========================
        st.subheader("Interpretasi Klaster")

        # Hitung mean global & per klaster dari data training
        cluster_centers = model.cluster_centers_
        cluster_means = pd.DataFrame(
            cluster_centers,
            columns=features
        )

        def range_kategori(rata):
            low_max = rata * 0.9
            high_min = rata * 1.1
            return low_max, high_min

        global_mean = cluster_means.mean()
        row = cluster_means.loc[cluster_result]

        range_wri = range_kategori(global_mean["WRI"])
        range_exposure = range_kategori(global_mean["Exposure"])
        range_vulnerability = range_kategori(global_mean["Vulnerability"])
        range_coping = range_kategori(global_mean["Lack of Coping Capabilities"])
        range_adaptive = range_kategori(global_mean["Lack of Adaptive Capacities"])

        def kategori(nilai, rata):
            if nilai > rata * 1.1:
                return "tinggi"
            elif nilai < rata * 0.9:
                return "rendah"
            else:
                return "sedang"

        # KATEGORI
        wri = kategori(row["WRI"], global_mean["WRI"])
        exposure = kategori(row["Exposure"], global_mean["Exposure"])
        vulnerability = kategori(row["Vulnerability"], global_mean["Vulnerability"])
        coping = kategori(
            row["Lack of Coping Capabilities"],
            global_mean["Lack of Coping Capabilities"]
        )
        adaptive = kategori(
            row["Lack of Adaptive Capacities"],
            global_mean["Lack of Adaptive Capacities"]
        )

        # LOGIKA LABEL
        if wri == "tinggi" and vulnerability == "tinggi" and coping == "tinggi":
            label = "High Risk – Low Resilience Regions"
            makna = (
                "Wilayah sangat rentan terhadap bencana dengan kapasitas "
                "respon dan adaptasi yang rendah."
            )
        elif wri == "rendah" and vulnerability == "rendah":
            label = "Low Risk – High Resilience Regions"
            makna = (
                "Wilayah relatif aman dengan sistem mitigasi dan adaptasi "
                "yang baik."
            )
        else:
            label = "Moderate Risk – Transitional Resilience Regions"
            makna = (
                "Wilayah memiliki tingkat risiko menengah dengan "
                "kapasitas adaptasi yang sedang."
            )

        st.markdown(f"**Label Klaster:** `{label}`")

        val_wri = input_data["WRI"]
        val_exposure = input_data["Exposure"]
        val_vulnerability = input_data["Vulnerability"]
        val_coping = input_data["Lack of Coping Capabilities"]
        val_adaptive = input_data["Lack of Adaptive Capacities"]

        st.markdown("**Karakteristik Utama (berdasarkan centroid klaster):**")
        st.markdown(f"""
        - **WRI:** {wri}  
        Rata-rata nilai WRI pada klaster ini adalah **{row['WRI']:.2f}**,    
        _(Rendah < {range_wri[0]:.2f} | Sedang {range_wri[0]:.2f}–{range_wri[1]:.2f} | Tinggi > {range_wri[1]:.2f})_  
        *(Nilai input pengguna: {val_wri:.2f})*

        - **Exposure:** {exposure}  
        Rata-rata nilai Exposure pada klaster ini adalah **{row['Exposure']:.2f}**, 
        _(Rendah < {range_exposure[0]:.2f} | Sedang {range_exposure[0]:.2f}–{range_exposure[1]:.2f} | Tinggi > {range_exposure[1]:.2f})_  
        *(Nilai input pengguna: {val_exposure:.2f})*

        - **Vulnerability:** {vulnerability}  
        Rata-rata nilai Vulnerability pada klaster ini adalah **{row['Vulnerability']:.2f}**,  
        _(Rendah < {range_vulnerability[0]:.2f} | Sedang {range_vulnerability[0]:.2f}–{range_vulnerability[1]:.2f} | Tinggi > {range_vulnerability[1]:.2f})_  
        *(Nilai input pengguna: {val_vulnerability:.2f})*

        - **Coping Capacity:** {coping}  
        Rata-rata nilai Coping Capacity pada klaster ini adalah **{row['Lack of Coping Capabilities']:.2f}**,  
        _(Rendah < {range_coping[0]:.2f} | Sedang {range_coping[0]:.2f}–{range_coping[1]:.2f} | Tinggi > {range_coping[1]:.2f})_  
        *(Nilai input pengguna: {val_coping:.2f})*

        - **Adaptive Capacity:** {adaptive}  
        Rata-rata nilai Adaptive Capacity pada klaster ini adalah **{row['Lack of Adaptive Capacities']:.2f}**,  
        _(Rendah < {range_adaptive[0]:.2f} | Sedang {range_adaptive[0]:.2f}–{range_adaptive[1]:.2f} | Tinggi > {range_adaptive[1]:.2f})_  
        *(Nilai input pengguna: {val_adaptive:.2f})*
        """)
        
        st.markdown(f"**Makna:** {makna}")

        # ===============================
        # Rekomendasi Kebijakan Wilayah
        # ===============================
        st.subheader("📌 Rekomendasi Kebijakan Wilayah")

        def rekomendasi_wilayah(cluster_label, input_data):
            rekomendasi = []

            if cluster_label == "High Risk – Low Resilience Regions":
                rekomendasi.append(
                    "Wilayah perlu menjadi prioritas utama dalam penguatan "
                    "mitigasi bencana dan ketahanan wilayah."
                )

            elif cluster_label == "Moderate Risk – Transitional Resilience Regions":
                rekomendasi.append(
                    "Wilayah berada pada fase transisi dan memerlukan "
                    "intervensi terarah untuk menurunkan risiko."
                )

            else:
                rekomendasi.append(
                    "Wilayah relatif aman, fokus pada pemeliharaan ketahanan "
                    "dan sistem monitoring risiko."
                )

            if input_data["Exposure"] > 15:
                rekomendasi.append(
                    "Paparan bencana tinggi → perlu penguatan tata ruang "
                    "dan sistem peringatan dini."
                )

            if input_data["Vulnerability"] > 50:
                rekomendasi.append(
                    "Kerentanan tinggi → perlu peningkatan program sosial "
                    "dan kesehatan."
                )

            if input_data["Lack of Coping Capabilities"] > 60:
                rekomendasi.append(
                    "Kapasitas penanggulangan rendah → perlu penguatan "
                    "institusi dan pelatihan kebencanaan."
                )

            if input_data["Lack of Adaptive Capacities"] > 50:
                rekomendasi.append(
                    "Kemampuan adaptasi rendah → perlu investasi "
                    "infrastruktur dan kebijakan adaptasi iklim."
                )

            return rekomendasi

        rekomendasi = rekomendasi_wilayah(label, input_data)

        for i, r in enumerate(rekomendasi, 1):
            st.markdown(f"**{i}.** {r}")


# app_prediksi_Klasifikasi_wilayah()
