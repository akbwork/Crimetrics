import pandas as pd
import folium
from folium.plugins import MarkerCluster, TimestampedGeoJson
import streamlit as st
from streamlit_folium import st_folium
from prophet import Prophet
import plotly.express as px
import json

# ---------- Load and Preprocess Data ---------- #
@st.cache_data

def load_data():
    df = pd.read_csv("crime_type_prediction_dataset_rajasthan.csv")
    df['datetime'] = pd.to_datetime(df['date_of_crime'] + ' ' + df['time_of_crime'])
    df['week'] = df['datetime'].dt.to_period('W').apply(lambda r: r.start_time)
    return df

df = load_data()

# Sample coordinates (expand as needed)
district_coords = {
    "Ajmer": (26.4499, 74.6399),
    "Alwar": (27.55299, 76.6346),
    "Barmer": (25.75, 71.4167),
    "Beawar": (26.1012, 74.3200),
    "Bikaner": (28.0229, 73.3119),
    "Dausa": (26.8938, 76.3375),
    "Jaipur": (26.9124, 75.7873),
    "Jodhpur": (26.2389, 73.0243),
    "Pali": (25.7725, 73.3234),
    "Salumbar": (24.1366, 74.0524),
    "Sawai Madhopur": (26.0370, 76.3560),
    "Udaipur": (24.5854, 73.7125)
}

coords_df = pd.DataFrame.from_dict(district_coords, orient='index', columns=['lat', 'lon']).reset_index()
coords_df.rename(columns={'index': 'district'}, inplace=True)

# ---------- Sidebar Filters ---------- #
st.sidebar.header("Filters")
category_filter = st.sidebar.multiselect("Select Crime Categories", df['crime_category'].unique(), default=df['crime_category'].unique())
date_range = st.sidebar.date_input("Select Date Range", [df['datetime'].min().date(), df['datetime'].max().date()])
district_filter = st.sidebar.multiselect("Select District(s)", df['district'].unique(), default=df['district'].unique())

# ---------- Filtered Data ---------- #
df_filtered = df[(df['crime_category'].isin(category_filter)) &
                 (df['district'].isin(district_filter)) &
                 (df['datetime'].dt.date >= date_range[0]) &
                 (df['datetime'].dt.date <= date_range[1])]

# ---------- Map Visualization ---------- #
st.title("🔎 Rajasthan Crime Hotspot Map")
crime_counts = df_filtered.groupby(['district', 'crime_category']).size().reset_index(name='count')
crime_summary = crime_counts.groupby('district').agg({"count": "sum"}).reset_index().rename(columns={'count': 'crime_count'})
map_data = pd.merge(crime_summary, coords_df, on='district', how='left').dropna(subset=['lat', 'lon'])

# Create tooltip text with category-wise breakdown
tooltip_data = crime_counts.groupby('district').apply(
    lambda x: "<br>".join([f"{row['crime_category']}: {row['count']}" for _, row in x.iterrows()])
).reset_index().rename(columns={0: 'tooltip'})

map_data = pd.merge(map_data, tooltip_data, on='district', how='left')

if crime_counts.empty:
    st.warning("No crimes found for selected filters. Try broadening your criteria.")
else:
    m = folium.Map(location=[26.9, 75.8], zoom_start=6, tiles="CartoDB positron")
    for _, row in map_data.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=max(row['crime_count'] * 0.5, 4),
            popup=folium.Popup(f"<b>{row['district']}</b><br>Total: {row['crime_count']}<br>{row['tooltip']}", max_width=300),
            color='crimson', fill=True, fill_opacity=0.7
        ).add_to(m)

    # Optional: Show GeoJSON borders
    if st.checkbox("Show District Boundaries"):
        try:
            with open("rajasthan_districts.geojson", "r") as f:
                geojson_data = json.load(f)
            folium.GeoJson(geojson_data, name="geojson").add_to(m)
        except FileNotFoundError:
            st.warning("GeoJSON file for Rajasthan districts not found. Please add it as 'rajasthan_districts.geojson'.")

    st_data = st_folium(m, width=900, height=600)

# ---------- Weekly Crime Trend Bar Chart ---------- #
st.header("Weekly Crime Trends")
weekly_trends = df_filtered.groupby(['week', 'district']).size().reset_index(name='crimes')
if not weekly_trends.empty:
    bar_chart = px.bar(weekly_trends, x='week', y='crimes', color='district', title="Weekly Crime Count by District")
    st.plotly_chart(bar_chart, use_container_width=True)
else:
    st.info("No weekly data to display based on the filters.")

# ---------- Forecasting ---------- #
st.header("Crime Forecasting")
selected_district = st.selectbox("Select District for Forecasting", df_filtered['district'].unique())
selected_category = st.selectbox("Select Crime Category", df_filtered['crime_category'].unique())

forecast_df = df_filtered[(df_filtered['district'] == selected_district) &
                          (df_filtered['crime_category'] == selected_category)]

ts = forecast_df.groupby('week').size().reset_index(name='y')
ts.rename(columns={'week': 'ds'}, inplace=True)

if len(ts) >= 2:
    model = Prophet()
    model.fit(ts)
    future = model.make_future_dataframe(periods=6, freq='W')
    forecast = model.predict(future)

    fig = px.line(forecast, x='ds', y='yhat', title=f"Forecast for {selected_category} in {selected_district}")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Police Patrol Recommendation ---------- #
    st.subheader("Patrol Suggestion")
    top_weeks = forecast.sort_values(by='yhat', ascending=False).head(3)[['ds', 'yhat']]
    st.markdown("Based on the forecast, increase patrol during the following weeks:")
    for _, row in top_weeks.iterrows():
        st.markdown(f"- **{row['ds'].date()}** (Estimated crimes: **{int(row['yhat'])}**) in **{selected_district}**")
else:
    st.warning("Not enough data for forecasting this selection.")

# ---------- Optional: Animated Time Map ---------- #
st.header("Animated Crime Trend Map")
if st.checkbox("Enable Time Animation"):
    time_data = df_filtered[df_filtered['district'].isin(district_coords.keys())].copy()
    time_data['timestamp'] = time_data['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    features = []
    for _, row in time_data.iterrows():
        lat, lon = district_coords[row['district']]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "time": row['timestamp'],
                "popup": f"{row['district']} - {row['crime_category']}"
            }
        })

    time_map = folium.Map(location=[26.9, 75.8], zoom_start=6)
    TimestampedGeoJson({
        "type": "FeatureCollection",
        "features": features,
    }, period="PT24H", add_last_point=True).add_to(time_map)

    st_folium(time_map, width=900, height=600)
