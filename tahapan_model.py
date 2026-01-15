import streamlit as st

def Tahapan_model():
    st.write("**Tahapan Model Lachine Learning**")
    
    pilihan = st.selectbox(
        "Pilih Jenis Tahapan Lachine Learning",
        ["klasifikasi", "Klastering"]
    )

    if pilihan == "klasifikasi":
        Langkah_model_Klasifikasi()
    elif pilihan == "Klastering":
        Langkah_model_Klastering()


def Langkah_model_Klasifikasi():
    import streamlit as st

    st.title("Langkah-Langkah Model Logistic Regression")

    st.markdown("""
    Berdasarkan hasil komparasi model klasifikasi yang telah dilakukan, dapat
    dilihat bahwa **model terbaik berdasarkan nilai akurasi, presisi, recall,
    F1-score, dan ROC AUC adalah Regresi Logistik**.

    Oleh karena itu, penting untuk memahami bagaimana alur matematis model Regresi Logistik bekerja dalam melakukan proses klasifikasi, karena model ini memodelkan log-odds suatu kejadian sebagai kombinasi linier dari variabel prediktor dan kemudian mengubahnya menjadi probabilitas untuk klasifikasi biner (Logistic Regression, Wikipedia; https://en.wikipedia.org/wiki/Logistic_regression .
    """)
    
    # ===============================
    # 1. Pendefinisian Variabel Respon
    # ===============================
    st.subheader("1️. Pendefinisian Variabel Respon (Bernoulli)")

    st.markdown(r"""
    Regresi logistik digunakan ketika variabel respon bersifat **dikotomi**, 
    yaitu hanya memiliki dua kemungkinan nilai.

    Secara matematis, variabel respon didefinisikan sebagai:

    $$Y_i \in \{0,1\}$$

    dengan keterangan:
    - $Y_i = 1$ → kejadian terjadi (**sukses**)
    - $Y_i = 0$ → kejadian tidak terjadi (**gagal**)

    Karena bersifat biner, maka variabel respon $Y_i$ mengikuti distribusi **Bernoulli**:

    $$Y_i \sim \text{Bernoulli}(p_i)$$

    di mana:

    $$p_i = P(Y_i = 1 \mid \mathbf{x}_i)$$

    merupakan probabilitas bahwa observasi ke-$i$ mengalami kejadian (kelas positif)
    berdasarkan vektor fitur $\mathbf{x}_i$.
    
    ---
    """)

    
    # ===============================
    # 2. Spesifikasi Model Logistik
    # ===============================
    st.subheader("2️. Spesifikasi Model Logistik (Link Function)")

    st.markdown(r"""
    Model regresi logistik menghubungkan probabilitas kejadian dengan variabel
    prediktor melalui **fungsi logit**.

    Fungsi logit didefinisikan sebagai:

    $$\text{logit}(p_i) = \ln\left(\frac{p_i}{1 - p_i}\right)$$

    Dengan menggunakan fungsi logit, hubungan antara probabilitas kejadian dan
    prediktor dapat dimodelkan secara linier.

    Model linier dari regresi logistik dituliskan sebagai:

    $$
    \ln\left(\frac{p_i}{1 - p_i}\right)
    = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_k x_{ik}
    $$

    di mana:
    - **β₀** adalah *intersep*
    - **β₁, β₂, …, βₖ** adalah *koefisien regresi*
    - **xᵢ₁, xᵢ₂, …, xᵢₖ** merupakan nilai fitur ke-k untuk observasi ke-i
    
    ---
    """)

    st.markdown(r"""
    ### 3️. Fungsi Probabilitas (Inverse Logit)

    Dari model logit yang telah dibentuk, diperoleh probabilitas terjadinya suatu kejadian sebagai berikut:

    $$
    p_i = \frac{e^{\beta_0 + \sum_{j=1}^{k} \beta_j x_{ij}}}
    {1 + e^{\beta_0 + \sum_{j=1}^{k} \beta_j x_{ij}}}
    $$

    atau dapat dituliskan dalam bentuk fungsi sigmoid:

    $$
    p_i = \frac{1}{1 + e^{-\eta_i}}
    $$

    dengan:

    $$
    \eta_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_k x_{ik}
    $$
    
    ---
    """)
    

    st.markdown(r"""
    ### 4️. Penyusunan Fungsi Likelihood

    Karena variabel respon $Y_i$ mengikuti distribusi **Bernoulli**, maka fungsi likelihood dari model regresi logistik dapat dituliskan sebagai:

    $$
    L(\boldsymbol{\beta}) = \prod_{i=1}^{n} p_i^{y_i} (1 - p_i)^{1 - y_i}
    $$

    di mana:
    - $p_i = P(Y_i = 1 \mid \mathbf{x}_i)$
    - $y_i \in \{0,1\}$

    ---

    ### 5️. Fungsi Log-Likelihood

    Untuk mempermudah proses estimasi parameter, fungsi likelihood ditransformasikan ke dalam bentuk **log-likelihood**, sehingga diperoleh:

    $$
    \ell(\boldsymbol{\beta}) =
    \sum_{i=1}^{n}
    \left[
    y_i \ln(p_i) + (1 - y_i)\ln(1 - p_i)
    \right]
    $$

    Dengan mensubstitusikan $p_i$ dari fungsi logistik (sigmoid), maka fungsi log-likelihood menjadi **fungsi nonlinier terhadap parameter** $\boldsymbol{\beta}$.
    """)

    st.markdown(r"""
    ---
    ### 6️. Estimasi Parameter (Maximum Likelihood Estimation)

    Parameter $\boldsymbol{\beta}$ diperoleh dengan cara memaksimumkan fungsi log-likelihood, yaitu dengan menyelesaikan persamaan:

    $$
    \frac{\partial \ell(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = 0
    $$

    Namun, karena fungsi log-likelihood regresi logistik **tidak memiliki solusi analitik tertutup (closed-form)**, maka proses estimasi dilakukan menggunakan **metode iteratif**.

    Metode optimasi yang umum digunakan antara lain:
    - **Newton–Raphson**
    - **Fisher Scoring**
    - **Iteratively Reweighted Least Squares (IRLS)**

    Bentuk umum pembaruan parameter pada metode Newton–Raphson adalah:

    $$
    \boldsymbol{\beta}^{(t+1)} =
    \boldsymbol{\beta}^{(t)} +
    \mathbf{I}^{-1}(\boldsymbol{\beta}^{(t)})
    \mathbf{U}(\boldsymbol{\beta}^{(t)})
    $$

    dengan:
    - $\mathbf{U}(\boldsymbol{\beta})$ adalah **vektor skor (gradien log-likelihood)**
    - $\mathbf{I}(\boldsymbol{\beta})$ adalah **matriks informasi Fisher**
    - $t$ menyatakan iterasi ke-$t$
    ---
    """)

    st.markdown(r"""
    ### 7️. Prediksi dan Klasifikasi

    Setelah parameter model regresi logistik diestimasi, langkah selanjutnya adalah melakukan **prediksi probabilitas** untuk setiap observasi.

    #### Prediksi Probabilitas

    Probabilitas bahwa observasi ke-$i$ termasuk ke dalam kelas $1$ diberikan oleh:

    $$
    \hat{p}_i = P(Y_i = 1 \mid \mathbf{x}_i)
    $$

    dengan $\hat{p}_i \in (0,1)$ merupakan hasil transformasi fungsi logit (sigmoid).

    #### Aturan Klasifikasi

    Berdasarkan nilai probabilitas tersebut, dilakukan klasifikasi menggunakan **nilai ambang (threshold)** $c$, sehingga prediksi kelas $\hat{Y}_i$ ditentukan sebagai berikut:

    $$
    \hat{Y}_i =
    \begin{cases}
    1, & \hat{p}_i \ge c \\
    0, & \hat{p}_i < c
    \end{cases}
    $$

    Nilai ambang yang umum digunakan adalah:

    $$
    c = 0.5
    $$

    Namun, nilai $c$ dapat disesuaikan tergantung pada tujuan analisis, seperti menyeimbangkan **presisi dan recall**, atau meminimalkan kesalahan klasifikasi tertentu.
    """)
    
# Langkah_model_Klasifikasi()

def Langkah_model_Klastering():
    import streamlit as st

    st.title("Langkah-Langkah Model Klastering Terbaik (K-Means)")

    st.markdown("""
    Berdasarkan hasil komparasi beberapa metode klastering yang telah dilakukan, 
    diperoleh bahwa **K-Means Clustering** menunjukkan performa paling stabil dan 
    konsisten berdasarkan nilai **Silhouette Coefficient**, 
    **Davies–Bouldin Index**, dan **Calinski–Harabasz Index**.  

    Selain memiliki kualitas klaster yang baik, K-Means juga lebih sederhana 
    dalam implementasi dan interpretasi, sehingga dipilih sebagai **model utama** 
    dalam proses pengelompokan wilayah.

    Berikut ini disajikan penjelasan matematis dan tahapan kerja algoritma K-Means yang dilakukan secara iteratif hingga mencapai kondisi konvergen (Afrizal Firdaus, 2020; https://medium.com/@afrizalfir/kmeans-clustering-dan-implementasinya-5e967dc604cf .
    """)

    st.markdown(r"""
    ### 1️⃣ Menentukan Jumlah Klaster ($k$)

    Pada tahap awal, ditentukan jumlah klaster $k$ yang akan digunakan. 
    Penentuan nilai $k$ dilakukan menggunakan **pendekatan Elbow Method** 
    serta diperkuat dengan evaluasi indeks validasi klaster internal.

    #### 🔹 Elbow Method
    Elbow Method dilakukan dengan menghitung **Within-Cluster Sum of Squares (WCSS)** 
    untuk berbagai nilai $k$. Nilai WCSS didefinisikan sebagai:

    $$
    WCSS = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k}
    \lVert \mathbf{x}_i - \boldsymbol{\mu}_k \rVert^2
    $$

    Nilai $k$ optimal ditentukan pada titik di mana penurunan WCSS mulai melambat 
    dan membentuk sudut siku (*elbow*), yang menunjukkan bahwa penambahan klaster 
    selanjutnya tidak memberikan peningkatan kualitas klaster yang signifikan.

    #### 🔹 Evaluasi Indeks Klaster
    Selain Elbow Method, pemilihan $k$ juga divalidasi menggunakan beberapa 
    metrik evaluasi internal, yaitu:
    - **Silhouette Score**
    - **Davies–Bouldin Index**
    - **Calinski–Harabasz Index**

    Tujuan pemilihan nilai $k$ adalah memperoleh klaster yang:
    - **Kompak secara internal**
    - **Terpisah dengan baik antar klaster**
    - **Stabil dan mudah diinterpretasikan**

    ---
    """)

    st.markdown(r"""
    ### 2️⃣ Inisialisasi Pusat Klaster

    Algoritma K-Means dimulai dengan memilih $k$ pusat klaster awal 
    $\mathbf{\mu}_1, \mathbf{\mu}_2, \dots, \mathbf{\mu}_k$ secara acak 
    dari ruang data.

    Setiap pusat klaster direpresentasikan sebagai vektor berdimensi sama 
    dengan data input.

    ---
    """)

    st.markdown(r"""
    ### 3️⃣ Menghitung Jarak Data ke Pusat Klaster

    Untuk setiap data $\mathbf{x}_i$, dihitung jaraknya terhadap seluruh 
    pusat klaster menggunakan **jarak Euclidean**:

    $$
    d_{ik} = \lVert \mathbf{x}_i - \mathbf{\mu}_k \rVert
    $$

    Jarak ini digunakan untuk menentukan klaster terdekat bagi setiap data.

    ---
    """)

    st.markdown(r"""
    ### 4️⃣ Menentukan Keanggotaan Klaster

    Setiap data $\mathbf{x}_i$ akan dimasukkan ke dalam klaster 
    dengan pusat terdekat, yaitu:

    $$
    \text{Cluster}(\mathbf{x}_i) =
    \arg \min_{k} \, d_{ik}
    $$

    Pada tahap ini, setiap data **hanya memiliki satu klaster** 
    (hard clustering).

    ---
    """)

    st.markdown(r"""
    ### 5️⃣ Memperbarui Pusat Klaster

    Setelah seluruh data dikelompokkan, pusat klaster baru dihitung 
    sebagai rata-rata dari seluruh data dalam klaster tersebut:

    $$
    \mathbf{\mu}_k =
    \frac{1}{|C_k|}
    \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i
    $$

    dengan $C_k$ merupakan himpunan data pada klaster ke-$k$.

    ---
    """)

    st.markdown(r"""
    ### 6️⃣ Iterasi hingga Konvergen

    Proses penugasan klaster dan pembaruan pusat klaster dilakukan 
    secara iteratif hingga memenuhi salah satu kondisi berikut:
    - Tidak ada perubahan keanggotaan klaster
    - Perubahan pusat klaster sangat kecil
    - Jumlah iterasi maksimum tercapai

    ---
    """)

    st.markdown(r"""
    ### 7️⃣ Hasil Klaster Akhir

    Hasil akhir dari algoritma K-Means berupa:
    - Label klaster untuk setiap wilayah
    - Pusat klaster sebagai representasi karakteristik wilayah
    - Rata-rata indikator pada setiap klaster untuk keperluan interpretasi

    Model ini kemudian digunakan dalam tahap **prediksi klaster wilayah baru** 
    berdasarkan indikator yang dimasukkan oleh pengguna.
    """)


# Langkah_model_Klastering()
