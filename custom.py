import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import KMeans

#Title
st.set_page_config(page_title="Customer Segmentation",layout="centered")


#page styling
st.title("🛍️ customer segmentation")
st.write("Hello I am Abhinandan Shukla")

#Data load

df=pd.read_csv("CustomerData.csv")

#Data preview
with st.expander("📈Data Preview"):
    st.dataframe(df)
x=df[["Age","Income"]]
k=st.slider("Enter your cluster",min_value=1,max_value=6)
model=KMeans(n_clusters=k,random_state=42,n_init=10)
df["cluster"]=model.fit_predict(x)
st.subheader("cluster data")
st.dataframe(df)



st.subheader("cluster center")
centers=pd.DataFrame(model.cluster_centers_,columns=["Age","Income"])
st.dataframe(df)


st.subheader("📋 Customer Dataset")
fig,ax=plt.subplots(figsize=(6,3))
scatter=ax.scatter(df["Age"],df["Income"],c=df["cluster"],cmap="viridis")
ax.scatter( model.cluster_centers_[:,0],model.cluster_centers_[:,1],marker="*",color="red",s=100,label="Centroids")

ax.set_title("K-Means Customer Clusters")
ax.set_xlabel("AGE")
ax.set_ylabel("INCOME")
ax.grid(True)
ax.legend()
st.pyplot(fig)