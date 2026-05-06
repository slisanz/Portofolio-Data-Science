from __future__ import annotations

import pandas as pd


def folium_points(df: pd.DataFrame, sample: int = 5000):
    import folium

    s = df.sample(min(len(df), sample), random_state=0)
    centre = [s["lat"].mean(), s["lon"].mean()]
    m = folium.Map(location=centre, zoom_start=11, tiles="cartodbpositron")
    for _, r in s.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=2,
            color="#cb4154" if r.get("is_lagging", 0) else "#2a9d8f",
            fill=True,
            fill_opacity=0.6,
            weight=0,
        ).add_to(m)
    return m


def pydeck_hex(df: pd.DataFrame, radius: int = 80):
    import pydeck as pdk

    layer = pdk.Layer(
        "HexagonLayer",
        df[["lon", "lat"]].rename(columns={"lon": "lng"}),
        get_position="[lng, lat]",
        radius=radius,
        elevation_scale=8,
        elevation_range=[0, 1500],
        extruded=True,
        pickable=True,
    )
    view = pdk.ViewState(
        longitude=float(df["lon"].mean()),
        latitude=float(df["lat"].mean()),
        zoom=11,
        pitch=45,
    )
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/light-v9",
    )


def commune_bar(equity_df: pd.DataFrame, top: int = 20):
    import plotly.express as px

    d = equity_df.sort_values("equity_index").head(top).reset_index()
    fig = px.bar(
        d,
        x="equity_index",
        y="nom_com",
        orientation="h",
        hover_data=["coverage", "pm_load", "coll_lag", "n_buildings"],
        title=f"Bottom {top} communes by equity index",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig
