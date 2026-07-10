import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="Customer Segmentation",
    layout="centered"
)

st.title("🛍️ Customer Segmentation")
st.write("Upload a CSV file to perform customer segmentation.")


uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)


def load_csv(file):
    """Safely load and validate the uploaded CSV file."""

    try:
        file.seek(0)
        dataframe = pd.read_csv(file)

    except UnicodeDecodeError:
        file.seek(0)
        dataframe = pd.read_csv(file, encoding="latin-1")

    return dataframe


if uploaded_file is not None:

    try:
        df = load_csv(uploaded_file)

        # Remove extra spaces from column names
        df.columns = df.columns.str.strip()

        if df.empty:
            st.error("The uploaded CSV file is empty.")
            st.stop()

        required_columns = ["Age", "Income"]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

            st.info(
                "Available columns: "
                + ", ".join(map(str, df.columns))
            )

            st.stop()

        # Convert columns to numeric values
        for column in required_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        invalid_rows = df[required_columns].isnull().any(axis=1).sum()

        if invalid_rows > 0:
            st.warning(
                f"{invalid_rows} rows containing invalid Age or Income "
                "values were removed."
            )

        df = df.dropna(subset=required_columns)

        if df.empty:
            st.error(
                "No valid numeric values were found in Age and Income."
            )
            st.stop()

        st.success("CSV file uploaded successfully.")

        with st.expander("📈 Data Preview", expanded=True):
            st.dataframe(df)

        maximum_clusters = min(6, len(df))

        if maximum_clusters < 1:
            st.error("The dataset does not contain enough rows.")
            st.stop()

        k = st.slider(
            "Select number of clusters",
            min_value=1,
            max_value=maximum_clusters,
            value=min(3, maximum_clusters)
        )

        x = df[["Age", "Income"]]

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        df["Cluster"] = model.fit_predict(x)

        st.subheader("📊 Clustered Data")
        st.dataframe(df)

        st.subheader("🎯 Cluster Centers")

        centers = pd.DataFrame(
            model.cluster_centers_,
            columns=["Age", "Income"]
        )

        centers.index.name = "Cluster"
        st.dataframe(centers)

        st.subheader("📋 Customer Segmentation Graph")

        fig, ax = plt.subplots(figsize=(8, 5))

        scatter = ax.scatter(
            df["Age"],
            df["Income"],
            c=df["Cluster"],
            cmap="viridis"
        )

        ax.scatter(
            model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            marker="*",
            color="red",
            s=200,
            label="Centroids"
        )

        ax.set_title("K-Means Customer Clusters")
        ax.set_xlabel("Age")
        ax.set_ylabel("Income")
        ax.grid(True)
        ax.legend()

        fig.colorbar(
            scatter,
            ax=ax,
            label="Cluster"
        )

        st.pyplot(fig)

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file contains no data.")

    except pd.errors.ParserError:
        st.error(
            "The uploaded file has an invalid CSV format."
        )

    except ValueError as error:
        st.error(f"Invalid data: {error}")

    except Exception as error:
        st.error(f"Unable to process the CSV file: {error}")

else:
    st.info(
        "Upload a CSV file containing Age and Income columns."
    )