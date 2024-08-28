import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

st.title("Customer Segmentation using K-Means Clustering")

excel_file = 'customer_purchase_behaviour.xlsx'
df = pd.read_excel(excel_file)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Age', 'Purchase_Amount', 'Review_Rating']])


# Load the trained K-means model and scaler if they exist, otherwise train a new one
try:
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    # st.write("Loaded saved model and scaler.")
except:
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    # st.write("Trained a new model and saved it.")

# Cluster the original data
df['Cluster'] = kmeans.predict(scaled_data)

# Sidebar for user input
st.sidebar.header('Customer Data Input')
age = st.sidebar.slider('Age', 18, 70, 30)
purchase_amount = st.sidebar.slider('Purchase Amount', 0, 50000, 10000)
review_rating = st.sidebar.slider('Review Rating', 0.0, 5.0, 4.0)


new_data = np.array([[age, purchase_amount, review_rating]])

# Scale new data point
scaled_new_data = scaler.transform(new_data)

# Predict the cluster for the new data point
new_data_cluster = kmeans.predict(scaled_new_data)

# Plotting the clusters and new data point
fig, ax = plt.subplots()

# Plot each cluster with a different color
colors = ['red', 'blue', 'green']
for i in range(3):
    ax.scatter(df[df['Cluster'] == i]['Age'], df[df['Cluster'] == i]['Purchase_Amount'],
               s=100, c=colors[i], label=f'Cluster {i+1}')

# Plot the new data point
ax.scatter(new_data[0, 0], new_data[0, 1], s=200, c='purple', label='New Data Point', marker='X')

plt.title('Clusters of Customers')
plt.xlabel('Age')
plt.ylabel('Purchase Amount')
plt.legend()

# Display the plot
st.pyplot(fig)

# Show the predicted cluster for the new data point
st.header(f"The new data point belongs to cluster: {new_data_cluster[0] + 1}")
st.write("")
st.write("")
st.write("")
if new_data_cluster[0]==0:
    st.error("**Customer belongs to the Low-Spending Customers**")
    st.write("**For better sales:**")
    st.write("**Promotions & Discounts:** Offer targeted discounts to encourage larger purchases.")
    st.write("**Loyalty Programs:** Introduce coupon vouchers for repeat purchases.")
    st.write("**Product Bundling:** Create bundles to provide better value and increase total spend.")
    st.write("**Upselling:** Recommend complementary products at checkout.")

elif new_data_cluster[0]==1:
    st.info("**Customer belongs to the High-Spending Customers**")
    st.write("**For better sales:**")
    st.write("**Exclusive Offers:** Provide early access to new products and special deals.")
    st.write("**Premium Services:** Offer VIP services like free shipping or personalized shopping.")
    st.write("**Tailored Recommendations:** Use data to suggest premium products.")
    st.write("**Engagement:** Collect feedback to strengthen loyalty.")

else:
    st.success("**Customer belongs to the Medium-Spending Customers**")
    st.write("**For better sales:**")
    st.write("**Upselling & Upgrades:** Encourage upgrading to premium products.")
    st.write("**Tiered Loyalty Programs:** Reward spending more with additional perks.")
    st.write("**Bundle Deals:** Offer bundles that include both familiar and new products")
    st.write("**Exclusive Previews:** Give memebership cards for the repeated shopping.")