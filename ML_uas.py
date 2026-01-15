import streamlit as st
import pickle


def ML():
    st.write("**Lachine Learning**")
    
    pilihan = st.selectbox(
        "Pilih Jenis Lachine Learning",
        ["klasifikasi", "Klastering"]
    )

    if pilihan == "klasifikasi":
        ML_klasifikasi()
    elif pilihan == "Klastering":
        ML_Klastering()

def ML_klasifikasi():
    import streamlit as st
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        classification_report
    )

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    import matplotlib.pyplot as plt
    import seaborn as sns
    import altair as alt

    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except:
        xgb_available = False

    st.title("Machine Learning Klasifikasi Data Kesehatan Kejadian Batu Empedu")

    # =========================
    # Load Data
    # =========================
    df = pd.read_excel("data uas ML kesehatan.xlsx")

    # =========================
    # 1. Tampilan Awal Data
    # =========================
    st.subheader("1. Dataset yang digunakan")
    st.dataframe(df.head())


    # =========================
    # Target & Fitur (DIKUNCI)
    # =========================
    y = df["Y_Gallstone Status"]
    X = df.drop(columns=["Y_Gallstone Status"])

    # Ambil kolom numerik
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X_num = X[num_cols]

    # =========================
    # 3. Visualisasi Box Plot
    # =========================
    st.subheader("2. Visualisasi Box Plot")
    st.write("**Distribusi Variabel Numerik (Box Plot)**")

    import altair as alt

    # Ambil kolom numerik & exclude kolom tertentu
    exclude_cols = ["Y_Gallstone Status", "X2_Gender"]
    box_cols_all = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    box_cols_all = [col for col in box_cols_all if col not in exclude_cols]

    # Opsi tampilkan semua atau pilih manual
    show_all = st.checkbox("Tampilkan semua variabel", value=True)

    if show_all:
        selected_cols = box_cols_all
    else:
        selected_cols = st.multiselect(
            "Pilih variabel yang ingin ditampilkan:",
            options=box_cols_all,
            default=box_cols_all[:3]
        )

    # Guard clause (biar nggak error)
    if len(selected_cols) == 0:
        st.warning("Pilih minimal satu variabel untuk ditampilkan.")
    else:
        # Data long format
        box_df = df[selected_cols].melt(
            var_name="Variabel",
            value_name="Nilai"
        )

        # Box plot Altair
        boxplot = alt.Chart(box_df).mark_boxplot().encode(
            x=alt.X("Variabel:N", title="Variabel"),
            y=alt.Y("Nilai:Q", title="Nilai"),
            color=alt.Color("Variabel:N", legend=None),
            tooltip=["Variabel", "Nilai"]
        ).properties(
            height=400
        )

        st.altair_chart(boxplot, use_container_width=True)

    # =========================
    # 4. Normalisasi Min–Max
    # =========================
    st.subheader("3. Normalisasi Data dengan Min–Max Scaling")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_num)
    X_scaled = pd.DataFrame(X_scaled, columns=num_cols)


    # =========================
    # 4. Analisis Korelasi
    # =========================
    st.subheader("4. Analisis Korelasi antar Variabel")

    corr = X_scaled.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


    # =========================
    # 6. Train-Test Split
    # =========================
    st.subheader("5. Pembagian Data")

    test_size = st.slider("Proporsi data uji:", 0.1, 0.5, 0.3)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # =========================
    # 7. Pemilihan Model
    # =========================
    st.subheader("6. Pemilihan Model Klasifikasi")

    model_name = st.selectbox(
        "Pilih model:",
        ["Logistic Regression", "Random Forest", "KNN", "SVM"] +
        (["XGBoost"] if xgb_available else [])
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "Random Forest":
        n_estimators = st.slider("Jumlah trees:", 50, 300, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    elif model_name == "KNN":
        k = st.slider("Jumlah K:", 3, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)

    elif model_name == "SVM":
        model = SVC(probability=True)

    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    # =========================
    # Training & Evaluasi
    # =========================
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # =========================
    # 7. Evaluasi Model
    # =========================
    st.subheader("7️. Evaluasi Model")

    from sklearn.metrics import ConfusionMatrixDisplay

    # Hitung metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    # =========================
    # Simpan Model Logistic Regression
    # =========================
    if model_name == "Logistic Regression":
        model_bundle = {
            "model": model,
            "scaler": scaler,
            "features": X_num.columns.tolist()
        }

        with open("logistic_regression_gallstone.pkl", "wb") as f:
            pickle.dump(model_bundle, f)

        # st.success("✅ Model Logistic Regression berhasil disimpan sebagai logistic_regression_gallstone.pkl")

    # Layout 2 kolom
    col1, col2 = st.columns([2, 1])

    # ===== KIRI: Confusion Matrix =====
    with col1:
        st.markdown("### Confusion Matrix")

        fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_test, y_pred)
        )
        disp.plot(cmap="Blues", ax=ax_cm, colorbar=True)
        ax_cm.set_title("")
        st.pyplot(fig_cm)

    # ===== KANAN: METRIK =====
    with col2:
        st.markdown("### Metrik Evaluasi")

        st.metric("Akurasi", f"{acc*100:.2f}%")
        st.metric("Precision", f"{prec*100:.2f}%")
        st.metric("Recall", f"{rec*100:.2f}%")
        st.metric("F1-Score", f"{f1*100:.2f}%")
        st.metric("ROC-AUC", f"{roc*100:.2f}%")

    # =========================
    # Evaluasi Semua Model (Untuk Tabel Perbandingan)
    # =========================
    all_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True)
    }

    if xgb_available:
        all_models["XGBoost"] = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss"
        )

    metrics_list = []

    for name, mdl in all_models.items():
        mdl.fit(X_train, y_train)
        y_pred_all = mdl.predict(X_test)

        if hasattr(mdl, "predict_proba"):
            y_proba_all = mdl.predict_proba(X_test)[:, 1]
            roc_all = roc_auc_score(y_test, y_proba_all)
        else:
            roc_all = np.nan

        metrics_list.append({
            "Model": name,
            "Akurasi": accuracy_score(y_test, y_pred_all),
            "Precision": precision_score(y_test, y_pred_all),
            "Recall": recall_score(y_test, y_pred_all),
            "F1-Score": f1_score(y_test, y_pred_all),
            "ROC-AUC": roc_all
        })

    metrics_df = pd.DataFrame(metrics_list)


    # =========================
    # Tambah Kolom Rata-rata
    # =========================
    metric_cols = ["Akurasi", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    metrics_df["Rata-rata"] = metrics_df[metric_cols].mean(axis=1)

    # =========================
    # Tabel Perbandingan Model
    # =========================
    st.markdown("### 📊 Perbandingan Performa Semua Model")

    metrics_df_display = metrics_df.copy()
    metrics_df_display.iloc[:, 1:] = metrics_df_display.iloc[:, 1:] * 100

    st.dataframe(
        metrics_df_display.style
        .format({
            "Akurasi": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1-Score": "{:.2f}%",
            "ROC-AUC": "{:.2f}%",
            "Rata-rata": "{:.2f}%"
        })
        .highlight_max(
            subset=["Akurasi", "Precision", "Recall", "F1-Score", "ROC-AUC", "Rata-rata"],
            color="lightgreen"
        ),
        use_container_width=True
    )

    best_model = metrics_df.loc[metrics_df["Rata-rata"].idxmax(), "Model"]
    st.success(
        f"🏆 Model dengan performa rata-rata tertinggi adalah **{best_model}**"
    )

    # =========================
    # 8. Feature Importance
    # =========================
    st.subheader("8️. Feature Importance")

    importance_df = None

    # ===== Logistic Regression =====
    if model_name == "Logistic Regression":
        coef = model.coef_[0]
        importance_df = pd.DataFrame({
            "Fitur": X_num.columns,
            "Importance": np.abs(coef)
        })

    # ===== Random Forest =====
    elif model_name == "Random Forest":
        importance_df = pd.DataFrame({
            "Fitur": X_num.columns,
            "Importance": model.feature_importances_
        })

    # ===== XGBoost =====
    elif model_name == "XGBoost":
        importance_df = pd.DataFrame({
            "Fitur": X_num.columns,
            "Importance": model.feature_importances_
        })

    # ===== Model tanpa feature importance =====
    else:
        st.info(
            f"""
            🔍 **Feature Importance tidak tersedia**

            Model **{model_name}** tidak menyediakan nilai feature importance secara
            langsung karena mekanisme kerjanya berbeda:

            - **KNN** → prediksi berdasarkan kedekatan jarak antar data
            - **SVM** → keputusan berbasis hyperplane, bukan kontribusi fitur

            👉 Untuk analisis feature importance, disarankan menggunakan:
            **Logistic Regression, Random Forest, atau XGBoost**.
            """
        )

    # ===== TAMPILAN KIRI – KANAN =====
    if importance_df is not None:
        importance_df = importance_df.sort_values(
            by="Importance", ascending=False
        ).reset_index(drop=True)

        col1, col2 = st.columns([1, 2])

        # ===== KIRI: TABEL =====
        with col1:
            st.markdown("### Tabel Feature Importance")
            st.dataframe(
                importance_df.style.format({"Importance": "{:.4f}"}),
                use_container_width=True
            )

        # ===== KANAN: BAR CHART =====
        with col2:
            st.markdown("### Visualisasi Feature Importance")

            bar = alt.Chart(importance_df).mark_bar().encode(
                x=alt.X(
                    "Importance:Q",
                    title="Nilai Importance"
                ),
                y=alt.Y(
                    "Fitur:N",
                    sort="-x",
                    title="Fitur"
                ),
                tooltip=[
                    alt.Tooltip("Fitur:N"),
                    alt.Tooltip("Importance:Q", format=".4f")
                ]
            ).properties(
                height=400
            )

            st.altair_chart(bar, use_container_width=True)

    if importance_df is not None:
        # =========================
        # 9. Interpretasi Feature Importance
        # =========================
        st.subheader("9️. Interpretasi Feature Importance")

        top_feature = importance_df.iloc[0]
        low_feature = importance_df.iloc[-1]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔝 Fitur dengan Importance Tertinggi")
            st.markdown(
                f"**{top_feature['Fitur']}** (≈ {top_feature['Importance']:.3f})"
            )
            st.write(
                f"""
                Fitur ini memiliki pengaruh paling dominan dalam proses prediksi model.
                Perubahan pada **{top_feature['Fitur']}** sangat sensitif terhadap hasil
                klasifikasi.
                """
            )

        with col2:
            st.markdown("### 🔽 Fitur dengan Importance Terendah")
            st.markdown(
                f"**{low_feature['Fitur']}** (≈ {low_feature['Importance']:.3f})"
            )
            st.write(
                f"""
                Fitur ini memiliki kontribusi paling kecil terhadap keputusan model dan
                cenderung berperan minor dalam proses klasifikasi.
                """
            )
        
# ML_klasifikasi()

def ML_Klastering():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import skfuzzy as fuzz

    st.title("Analisis Klastering (Pengelombokan Negara berdasarkan Indikator Risiko Bencana)")

    # =========================
    # Load Data
    # =========================
    df = pd.read_excel("data uas ML Lingkungan.xlsx")

    # ===============================
    # 1. Data Head
    # ===============================
    st.subheader("Data Awal")
    st.dataframe(df.head())

    # Ambil data numerik
    num_df = df.select_dtypes(include=np.number)


    # =========================
    # 3. Visualisasi Box Plot
    # =========================
    st.subheader("Visualisasi Box Plot")
    st.write("**Distribusi Variabel Numerik (Box Plot)**")

    import altair as alt

    # Ambil kolom numerik
    box_cols_all = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Opsi tampilkan semua atau pilih manual
    show_all = st.checkbox("Tampilkan semua variabel", value=True)

    if show_all:
        selected_cols = box_cols_all
    else:
        selected_cols = st.multiselect(
            "Pilih variabel yang ingin ditampilkan:",
            options=box_cols_all,
            default=box_cols_all[:3]
        )

    # Guard clause
    if len(selected_cols) == 0:
        st.warning("Pilih minimal satu variabel untuk ditampilkan.")
    else:
        # Ubah ke long format
        box_df = df[selected_cols].melt(
            var_name="Variabel",
            value_name="Nilai"
        )

        # Box plot Altair
        boxplot = alt.Chart(box_df).mark_boxplot().encode(
            x=alt.X("Variabel:N", title="Variabel"),
            y=alt.Y("Nilai:Q", title="Nilai"),
            color=alt.Color("Variabel:N", legend=None),
            tooltip=["Variabel", "Nilai"]
        ).properties(
            height=400
        )

        st.altair_chart(boxplot, use_container_width=True)

    # ===============================
    # 4. Normalisasi
    # ===============================
    scaler = MinMaxScaler()
    X = scaler.fit_transform(num_df)

    # ===============================
    # Layout 2 Kolom
    # ===============================
    col1, col2 = st.columns([2, 1])

    # ===============================
    # KOLOM KIRI → ELBOW
    # ===============================
    with col1:
        st.subheader("Metode Elbow untuk Penentuan Jumlah Klaster")

        inertia = []
        K = range(2, 11)

        for k_elbow in K:
            kmeans = KMeans(n_clusters=k_elbow, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K, inertia, marker='o')
        ax.set_xlabel("Jumlah Klaster")
        ax.set_ylabel("Inertia")
        st.pyplot(fig)

    # ===============================
    # KOLOM KANAN → MODEL + EVALUASI
    # ===============================
    with col2:
        st.subheader("Pemilihan Model Klastering")

        model_choice = st.selectbox(
            "Pilih Metode Klastering",
            ["K-Means", "Fuzzy C-Means", "DBSCAN", "Hierarki", "Grid Based"]
        )

        k = st.slider("Jumlah Klaster", 2, 6, 3)

        # ----- K-MEANS -----
        if model_choice == "K-Means":
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X)

            # =========================
            # SIMPAN MODEL K-MEANS
            # =========================
            kmeans_bundle = {
                "model": model,
                "scaler": scaler,
                "features": num_df.columns.tolist(),
                "n_clusters": k,
                "cluster_centers": model.cluster_centers_
            }

            with open("kmeans_clustering_model.pkl", "wb") as f:
                pickle.dump(kmeans_bundle, f)

            st.success("✅ Model K-Means berhasil disimpan")

        # ----- FUZZY C-MEANS -----
        elif model_choice == "Fuzzy C-Means":
            try:
                import skfuzzy as fuzz
            except:
                st.error("Library scikit-fuzzy belum terinstall.")
                st.stop()

            # Training FCM
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                X.T, c=k, m=2, error=0.005, maxiter=1000
            )

            labels = np.argmax(u, axis=0)

            # Hitung cluster means (UNTUK INTERPRETASI SAJA)
            df_clustered = num_df.copy()
            df_clustered["Cluster"] = labels

            cluster_means = (
                df_clustered
                .groupby("Cluster")[num_df.columns]
                .mean()
            )

        # ----- DBSCAN -----
        elif model_choice == "DBSCAN":
            model = DBSCAN(eps=0.3, min_samples=5)
            labels = model.fit_predict(X)

        # ----- HIERARKI -----
        elif model_choice == "Hierarki":
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X)

        # ----- GRID BASED (pendekatan) -----
        else:
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X)

        df["Cluster"] = labels
        
        st.markdown("### Evaluasi Klastering")

        if len(set(labels)) > 1 and -1 not in set(labels):
            st.markdown(f"**Silhouette Score:** `{silhouette_score(X, labels):.3f}`")
            st.markdown(f"**Davies-Bouldin Index:** `{davies_bouldin_score(X, labels):.3f}`")
            st.markdown(f"**Calinski-Harabasz Index:** `{calinski_harabasz_score(X, labels):.2f}`")
        else:
            st.warning("Evaluasi tidak valid (klaster tunggal atau noise dominan)")


    st.markdown("### 📊 Perbandingan Evaluasi Klastering")

    cluster_range = range(2, 7)
    evaluation_results = []

    # =========================
    # 1. K-MEANS, HIERARKI, GRID
    # =========================
    k_based_models = {
        "K-Means": lambda k: KMeans(n_clusters=k, random_state=42),
        "Hierarki": lambda k: AgglomerativeClustering(n_clusters=k),
        "Grid Based": lambda k: KMeans(n_clusters=k, random_state=42)
    }

    for model_name_eval, model_fn in k_based_models.items():
        for k_eval in cluster_range:
            labels_eval = model_fn(k_eval).fit_predict(X)

            if len(set(labels_eval)) > 1:
                evaluation_results.append({
                    "Model": model_name_eval,
                    "Jumlah Klaster": k_eval,
                    "Silhouette": silhouette_score(X, labels_eval),
                    "Davies-Bouldin": davies_bouldin_score(X, labels_eval),
                    "Calinski-Harabasz": calinski_harabasz_score(X, labels_eval)
                })

    # =========================
    # 2. FUZZY C-MEANS
    # =========================
    for k_eval in cluster_range:
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X.T, c=k_eval, m=2, error=0.005, maxiter=1000
        )
        labels_eval = np.argmax(u, axis=0)

        if len(set(labels_eval)) > 1:
            evaluation_results.append({
                "Model": "Fuzzy C-Means",
                "Jumlah Klaster": k_eval,
                "Silhouette": silhouette_score(X, labels_eval),
                "Davies-Bouldin": davies_bouldin_score(X, labels_eval),
                "Calinski-Harabasz": calinski_harabasz_score(X, labels_eval)
            })

    # =========================
    # 3. DBSCAN (TANPA K)
    # =========================
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels_db = dbscan.fit_predict(X)

    valid_mask = labels_db != -1
    unique_clusters = set(labels_db[valid_mask])

    if len(unique_clusters) > 1:
        evaluation_results.append({
            "Model": "DBSCAN",
            "Jumlah Klaster": len(unique_clusters),
            "Silhouette": silhouette_score(X[valid_mask], labels_db[valid_mask]),
            "Davies-Bouldin": davies_bouldin_score(X[valid_mask], labels_db[valid_mask]),
            "Calinski-Harabasz": calinski_harabasz_score(X[valid_mask], labels_db[valid_mask])
        })

    eval_df = pd.DataFrame(evaluation_results)

    # =========================
    # Tambah Kolom Rata-rata (Normalized)
    # =========================
    eval_df_norm = eval_df.copy()

    # Invers Davies-Bouldin (semakin kecil semakin baik)
    eval_df_norm["DB_inv"] = 1 / eval_df_norm["Davies-Bouldin"]

    metric_cols = ["Silhouette", "Calinski-Harabasz", "DB_inv"]

    eval_df_norm["Rata-rata"] = eval_df_norm[metric_cols].mean(axis=1)

    # Gabungkan ke dataframe asli
    eval_df["Rata-rata"] = eval_df_norm["Rata-rata"]


    selected_k_eval = st.selectbox(
        "Pilih jumlah klaster:",
        sorted(eval_df["Jumlah Klaster"].unique())
    )

    filtered_eval_df = eval_df[eval_df["Jumlah Klaster"] == selected_k_eval]

    st.dataframe(
        filtered_eval_df
        .assign(**{"Rata-rata": eval_df.loc[filtered_eval_df.index, "Rata-rata"]})
        .sort_values("Rata-rata", ascending=False)
        .style
        .format({
            "Silhouette": "{:.5f}",
            "Davies-Bouldin": "{:.5f}",
            "Calinski-Harabasz": "{:.5f}",
            "Rata-rata": "{:.5f}"
        })
        .highlight_max(
            subset=["Silhouette", "Calinski-Harabasz", "Rata-rata"],
            color="lightgreen"
        )
        .highlight_min(
            subset=["Davies-Bouldin"],
            color="lightgreen"
        ),
        use_container_width=True
    )

    best_row = (
        eval_df
        .sort_values("Rata-rata", ascending=False)
        .iloc[0]
    )

    # =========================
    # Model Klastering Terbaik (Handling Tie)
    # =========================
    best_score = eval_df["Rata-rata"].max()

    best_models = eval_df[eval_df["Rata-rata"] == best_score]

    st.success("🏆 **Model Klastering Terbaik (Skor Rata-rata Tertinggi)**")

    for i, (_, row) in enumerate(best_models.iterrows(), start=1):
        st.markdown(
            f"""
            **Model Terbaik ke-{i}**

            - **Metode:** {row['Model']}
            - **Jumlah Klaster Optimal:** {int(row['Jumlah Klaster'])}
            - **Silhouette Score:** {row['Silhouette']:.5f}
            - **Davies-Bouldin Index:** {row['Davies-Bouldin']:.5f}
            - **Calinski-Harabasz Index:** {row['Calinski-Harabasz']:.5f}
            - **Skor Rata-rata:** {row['Rata-rata']:.5f}
            """
        )

    st.info(
        "Catatan: Lebih dari satu model memiliki skor rata-rata yang identik "
        "hingga lima angka di belakang koma, sehingga seluruh model tersebut "
        "ditampilkan sebagai model terbaik."
    )


    # ===============================
    # 8. Visualisasi Keanggotaan Klaster (PCA)
    # ===============================
    from sklearn.decomposition import PCA

    st.subheader("Visualisasi Keanggotaan Klaster (PCA)")

    # PCA ke 2 dimensi
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": labels.astype(str)
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10",
        ax=ax
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Visualisasi Klaster Menggunakan PCA")
    ax.legend(title="Cluster")

    st.pyplot(fig)

    # ===============================
    # 9. Jumlah & Anggota Klaster
    # ===============================
    st.subheader("Ringkasan Klaster")

    col_left, col_right = st.columns([2, 1])

    # ===============================
    # KIRI → Negara berdasarkan Klaster
    # ===============================
    with col_left:
        st.markdown("### 🔍 Negara berdasarkan Klaster")

        selected_cluster = st.selectbox(
            "Pilih Klaster:",
            sorted(df["Cluster"].unique())
        )

        negara_klaster = df[df["Cluster"] == selected_cluster]["Region"]

        st.markdown(
            f"**Jumlah negara di Klaster {selected_cluster}: "
            f"{len(negara_klaster)}**"
        )

        st.dataframe(
            negara_klaster
            .reset_index(drop=True)
            .to_frame(name="Region"),
            height=350
        )

    # ===============================
    # KANAN → Jumlah & Anggota Klaster
    # ===============================
    with col_right:
        st.markdown("### Jumlah & Anggota Klaster")

        cluster_count = (
            df["Cluster"]
            .value_counts()
            .rename_axis("Cluster")
            .reset_index(name="count")
            .sort_values("Cluster")
        )

        st.dataframe(
            cluster_count,
            height=180
        )

    # ===============================
    # Interpretasi Klaster
    # ===============================
    st.subheader("🧠 Interpretasi Klaster")

    cluster_means = df.groupby("Cluster")[num_df.columns].mean()

    # ===============================
    # Hitung Risk Score (agregat)
    # ===============================
    risk_score = cluster_means[
        [
            "WRI",
            "Exposure",
            "Vulnerability",
            "Lack of Coping Capabilities",
            "Lack of Adaptive Capacities"
        ]
    ].mean(axis=1)

    # ===============================
    # Urutkan Klaster dari Low → High Risk
    # ===============================
    sorted_clusters = risk_score.sort_values().index.tolist()
    n_clusters = len(sorted_clusters)

    # ===============================
    # Mapping Label Berdasarkan Ranking
    # ===============================
    label_map = {}

    for i, cluster_id in enumerate(sorted_clusters):
        if i < n_clusters / 3:
            label_map[cluster_id] = (
                "Low Risk – High Resilience Regions",
                "Wilayah relatif aman dengan sistem mitigasi dan adaptasi yang baik."
            )
        elif i < 2 * n_clusters / 3:
            label_map[cluster_id] = (
                "Moderate Risk – Transitional Resilience Regions",
                "Wilayah dengan risiko menengah dan kapasitas adaptasi yang sedang."
            )
        else:
            label_map[cluster_id] = (
                "High Risk – Low Resilience Regions",
                "Wilayah sangat rentan terhadap bencana dengan kapasitas respons dan adaptasi yang rendah."
            )

    # ===============================
    # Loop Interpretasi Klaster
    # ===============================
    for cluster_id in sorted_clusters:
        st.markdown(f"### 🔹 Klaster {cluster_id}")

        col_left, col_right = st.columns([2, 1])
        row = cluster_means.loc[cluster_id]

        # ===============================
        # Kategori relatif (informasi tambahan)
        # ===============================
        def kategori(nilai, rata):
            if nilai > rata:
                return "tinggi"
            elif nilai < rata:
                return "rendah"
            else:
                return "sedang"

        global_mean = cluster_means.mean()

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

        # ===============================
        # Ambil Label & Makna dari Ranking
        # ===============================
        label, makna = label_map[cluster_id]

        # ===============================
        # KIRI → INTERPRETASI
        # ===============================
        with col_left:
            st.markdown("**Ciri Utama:**")
            st.markdown(f"""
            - **WRI:** {wri}
            - **Exposure:** {exposure}
            - **Vulnerability:** {vulnerability}
            - **Coping Capacity:** {coping}
            - **Adaptive Capacity:** {adaptive}
            """)

            st.markdown(f"🧠 **Label Klaster:** 👉 `{label}`")
            st.markdown(f"**Makna:** {makna}")

        # ===============================
        # KANAN → STATISTIK MEAN
        # ===============================
        with col_right:
            st.markdown("**Rata-rata Indikator**")

            mean_df = (
                row
                .round(2)
                .reset_index()
                .rename(columns={"index": "Indikator", 0: "Mean"})
            )

            st.dataframe(mean_df, height=250)


# ML_Klastering()
