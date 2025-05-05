# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import folium
import plotly.express as px

# Page config
st.set_page_config(page_title="NYC Taxi Cluster Explorer", layout="wide")
st.title('🚖 NYC Yellow Taxi Pickup Zone Analysis')

# Load preprocessed data
cluster_centers = pd.read_csv("cluster_centers.csv")
cluster_stats = pd.read_csv("cluster_stats.csv")

# Weekday mapping
weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_name_to_number = {name: i for i, name in enumerate(weekday_names)}

# Time selectors
col1, col2 = st.columns(2)
with col1:
    selected_weekday_name = st.selectbox('📆 Select Day of the Week', weekday_names)
    selected_weekday = weekday_name_to_number[selected_weekday_name]
with col2:
    selected_hour = st.selectbox('🕒 Select Hour (0–23)', list(range(24)))

# Map interaction section
st.header('🗺 Click on the Map to Get Pickup Zone Recommendations')
st.markdown("👉 Click on any location on the map below. The system will recommend the nearest and most active zone based on your selected day and hour.")

# Initial map
m = folium.Map(location=[40.75, -73.98], zoom_start=12)
for _, row in cluster_centers.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=3,
        fill=True,
        fill_color='blue',
        color='blue',
        fill_opacity=0.7,
        popup=f"Cluster {row['cluster_id']}"
    ).add_to(m)

map_data = st_folium(m, height=500, width=700)

# Recommendation logic based on map click
if map_data.get('last_clicked') is not None:
    clicked_lat = map_data['last_clicked']['lat']
    clicked_lon = map_data['last_clicked']['lng']
    st.success(f'📍 You clicked: ({clicked_lat:.4f}, {clicked_lon:.4f})')

    # Find nearest cluster
    user_point = np.array([clicked_lon, clicked_lat])
    cluster_points = cluster_centers[['lon', 'lat']].to_numpy()
    distances = np.linalg.norm(cluster_points - user_point, axis=1)
    nearest_cluster_id = int(np.argmin(distances))
    nearest_row = cluster_centers[cluster_centers['cluster_id'] == nearest_cluster_id].iloc[0]

    # Find most active cluster at selected time
    time_filtered = cluster_stats[
        (cluster_stats['pickup_hour'] == selected_hour) &
        (cluster_stats['pickup_weekday'] == selected_weekday)
    ]
    if not time_filtered.empty:
        hottest_cluster_id = int(time_filtered.sort_values('trip_count', ascending=False).iloc[0]['cluster_id'])
        hottest_row = cluster_centers[cluster_centers['cluster_id'] == hottest_cluster_id].iloc[0]
    else:
        hottest_cluster_id = None
        hottest_row = None

    # Show recommendation results
    st.subheader("✅ Recommended Clusters")
    st.write(f"📌 Nearest Pickup Spot: **{nearest_cluster_id}** (lat: {nearest_row['lat']:.4f}, lon: {nearest_row['lon']:.4f})")
    if hottest_row is not None:
        st.write(f"🔥 Busiest Pickup Spot at that time: **{hottest_cluster_id}** (lat: {hottest_row['lat']:.4f}, lon: {hottest_row['lon']:.4f})")
    else:
        st.write("⚠️ No trip data found for the selected time.")

    # Map with highlighted results
    m2 = folium.Map(location=[clicked_lat, clicked_lon], zoom_start=12)
    for _, row in cluster_centers.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            fill=True,
            fill_color='blue',
            color='blue',
            fill_opacity=0.6,
            popup=f"Cluster {row['cluster_id']}"
        ).add_to(m2)

    folium.Marker(
        location=[clicked_lat, clicked_lon],
        popup='Your Click',
        icon=folium.Icon(color='black')
    ).add_to(m2)

    folium.Marker(
        location=[nearest_row['lat'], nearest_row['lon']],
        popup=f'Nearest Cluster {nearest_cluster_id}',
        icon=folium.Icon(color='green')
    ).add_to(m2)

    if hottest_row is not None:
        folium.Marker(
            location=[hottest_row['lat'], hottest_row['lon']],
            popup=f'Hottest Cluster {hottest_cluster_id}',
            icon=folium.Icon(color='red')
        ).add_to(m2)

    st_folium(m2, height=500, width=700)

# Animated hourly heatmap
st.header("🔥 Animated Pickup Spot Activity Heatmap (Hourly)")

# Merge location and stats
stats_with_location = cluster_stats.merge(cluster_centers, on='cluster_id')
filtered_day = stats_with_location[stats_with_location['pickup_weekday'] == selected_weekday]

fig = px.scatter_mapbox(
    filtered_day,
    lat="lat",
    lon="lon",
    size="trip_count",
    color="trip_count",
    animation_frame="pickup_hour",
    color_continuous_scale="YlOrRd",
    size_max=15,
    zoom=11,
    height=600,
    title=f"Hourly Pickup Spot Activity on {selected_weekday_name}"
)

fig.update_layout(mapbox_style="carto-positron")
st.plotly_chart(fig, use_container_width=True)

# Cluster bar chart
st.header("📊 View Trip Activity of a Specific Pickup Zone")
selected_cluster_id = st.selectbox("Select a Cluster ID", cluster_stats['cluster_id'].unique())
filtered = cluster_stats[cluster_stats['cluster_id'] == selected_cluster_id]

fig2 = px.bar(
    filtered,
    x='pickup_hour',
    y='trip_count',
    color='pickup_weekday',
    labels={'pickup_hour': 'Hour', 'trip_count': 'Trips', 'pickup_weekday': 'Weekday'},
    title=f"Cluster {selected_cluster_id} - Trip Volume by Hour and Weekday"
)
st.plotly_chart(fig2, use_container_width=True)

# Raw data table
with st.expander("🧮 Show Raw Cluster Stats Table"):
    st.dataframe(cluster_stats)
