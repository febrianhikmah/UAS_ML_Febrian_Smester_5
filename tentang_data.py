import streamlit as st

def about_dataset():
    st.write("**Tentang Dataset**")

    pilihan = st.selectbox(
        "Pilih Jenis Data",
        ["Kesehatan", "Lingkungan"]
    )

    # =======================
    # BARIS 1 : GAMBAR
    # =======================
    col1, col2 = st.columns([4, 6])

    with col1:
        if pilihan == "Kesehatan":
            st.image(
                "https://keslan.kemkes.go.id/img/bg-img/gambarartikel_1701676464_525951.jpg",
                caption="Penyakit Batu Empedu",
                use_container_width=True
            )
        else:
            st.image(
                "https://www.shutterstock.com/shutterstock/photos/2527822493/display_1500/stock-vector-international-day-for-disaster-risk-reduction-vector-illustration-volcano-eruption-floods-2527822493.jpg",
                caption="Risiko Bencana Lingkungan",
                use_container_width=True
            )

    with col2:
        if pilihan == "Kesehatan":
            st.subheader("Dataset Klasifikasi Penyakit Batu Empedu")
        else:
            st.subheader("Dataset Klastering Risiko Bencana")

    # =======================
    # BARIS 2 : TEKS (FULL WIDTH)
    # =======================
    if pilihan == "Kesehatan":
        st.markdown("""
https://archive.ics.uci.edu/dataset/1150/gallstone-1 .

Dataset klinis ini dikumpulkan dari Poliklinik Rawat Jalan Penyakit Dalam Rumah Sakit Ankara VM Medical Park dan mencakup data dari 319 individu pada periode Juni 2022 hingga Juni 2023, di mana 161 di antaranya didiagnosis menderita penyakit batu empedu. Dataset ini terdiri dari 38 fitur yang mencakup data demografis, bioimpedansi, dan data laboratorium, serta telah memperoleh persetujuan etik dari Komite Etik Rumah Sakit Kota Ankara (E2-23-4632).

Variabel demografis meliputi usia, jenis kelamin, tinggi badan, berat badan, dan indeks massa tubuh (BMI). Data bioimpedansi mencakup air tubuh total, air ekstraseluler dan intraseluler, massa otot dan lemak, protein, luas lemak viseral, serta lemak hati. Fitur laboratorium meliputi glukosa, kolesterol total, HDL, LDL, trigliserida, AST, ALT, ALP, kreatinin, laju filtrasi glomerulus (GFR), CRP, hemoglobin, dan vitamin D.

Dataset ini lengkap tanpa nilai hilang dan seimbang berdasarkan status penyakit, sehingga tidak memerlukan prapemrosesan tambahan. Dataset ini memberikan dasar yang kuat untuk pengembangan model machine learning dalam prediksi penyakit batu empedu.

Dalam Penelitian ini, variabel yang di gunakan sebagi berikut, Variabel respon (Y) adalah status penyakit batu empedu (Gallstone Status). Variabel prediktor meliputi usia, jenis kelamin, komorbiditas, indeks massa tubuh (BMI), tingkat lemak viseral, kadar glukosa darah, kolesterol total, serta kadar vitamin D.
        """)
    else:
        st.markdown("""
https://www.kaggle.com/datasets/tr1gg3rtrash/global-disaster-risk-index-time-series-dataset.

Dataset lingkungan ini digunakan untuk menggambarkan tingkat risiko dan kerentanan wilayah terhadap bencana dan tekanan lingkungan. Unit analisis dalam dataset ini adalah wilayah atau negara (Region), seperti Vanuatu dan Tonga.

Dataset ini memuat beberapa indeks komposit, yaitu World Risk Index (WRI), Exposure, dan Vulnerability. WRI merupakan indikator utama yang mencerminkan tingkat risiko suatu wilayah terhadap bencana, yang diperoleh dari kombinasi tingkat paparan dan kerentanan wilayah tersebut.

Variabel Exposure menggambarkan sejauh mana suatu wilayah terpapar terhadap bahaya alam seperti bencana geofisik dan hidrometeorologi. Vulnerability mencerminkan kondisi sosial, ekonomi, dan struktural yang memengaruhi kemampuan wilayah dalam menghadapi dampak bencana.

Variabel Susceptibility menunjukkan kondisi dasar masyarakat yang meningkatkan kerentanan. Lack of Coping Capabilities menggambarkan keterbatasan kemampuan jangka pendek wilayah dalam merespons bencana, sedangkan Lack of Adaptive Capacities mencerminkan keterbatasan kemampuan jangka panjang dalam beradaptasi terhadap perubahan lingkungan dan iklim.

Dataset ini dapat dimanfaatkan sebagai dasar analisis kuantitatif, klasterisasi wilayah risiko, serta pengembangan model machine learning untuk pemodelan dan prediksi risiko lingkungan.
        """)

