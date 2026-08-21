"""
Restaurant Insights & Rating Estimator
----------------------------------------
A simple two-page dashboard.

Page 1 - Restaurant Insights   : Explore the restaurant data with charts.
Page 2 - Rating Estimator      : Fill a small form and get an estimated rating.

HOW TO RUN
1) Put these files in the SAME folder:
      - app.py            (this file)
      - Dataset.csv        (the restaurant data used for the charts)
      - best_model.pkl     (the saved model from your notebook)
2) Install what's needed (only once):
      pip install streamlit pandas numpy plotly category_encoders scikit-learn
3) Start the app:
      streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------
# Page setup (must be the very first Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Restaurant Insights & Rating Estimator",
    page_icon="🍽️",
    layout="wide",
)

# ----------------------------------------------------------------------
# Look & feel — a bright, consistent colour palette used everywhere
# ----------------------------------------------------------------------
PALETTE = [
    "#FF6B6B", "#4ECDC4", "#FFD93D", "#6C5CE7", "#1DD3B0",
    "#F86624", "#2E86AB", "#F94892", "#43BCCD", "#F4A261",
]
px.defaults.color_discrete_sequence = PALETTE
px.defaults.template = "plotly_white"

st.markdown(
    """
    <style>
    .main-banner {
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        background: linear-gradient(90deg, #FF6B6B 0%, #F86624 35%, #FFD93D 70%, #4ECDC4 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .main-banner h1 { margin: 0; color: white; }
    .main-banner p { margin: 0.2rem 0 0 0; color: #fff4f4; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #eee;
        border-left: 6px solid #FF6B6B;
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div[data-testid="stMetricValue"],
    div[data-testid="stMetric"] * {
        color: #1a1a1a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Helpers to load the data & the saved model (cached so it's fast)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("Dataset.csv")
        return data
    except FileNotFoundError:
        return None


@st.cache_resource
def load_model():
    try:
        with open("best_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


df = load_data()
model = load_model()

# Known country codes used in this dataset, shown by country name so it's
# easy to pick without remembering numbers.
COUNTRY_OPTIONS = {
    "India": 1,
    "Australia": 14,
    "Brazil": 30,
    "Canada": 37,
    "Indonesia": 94,
    "New Zealand": 148,
    "Philippines": 162,
    "Qatar": 166,
    "Singapore": 184,
    "South Africa": 189,
    "Sri Lanka": 191,
    "Turkey": 208,
    "UAE": 214,
    "United Kingdom": 215,
    "United States": 216,
}
COUNTRY_CODE_TO_NAME = {v: k for k, v in COUNTRY_OPTIONS.items()}

CURRENCY_OPTIONS = [
    "Indian Rupees",
    "Dollar",
    "Pounds",
    "Botswana Pula",
    "Emirati Diram",
    "Turkish Lira",
    "Brazilian Real",
    "NewZealand",
    "Rand",
    "Sri Lankan Rupee",
    "Qatari Rial",
    "Indonesian Rupiah",
]

OTHER_LABEL = "Other (type it myself)"


@st.cache_data
def get_dropdown_options(data: pd.DataFrame):
    """Build the lists used to fill the select boxes, straight from the data."""
    cuisine_series = (
        data["Cuisines"].dropna().str.split(",").explode().str.strip()
    )
    cuisines = sorted(c for c in cuisine_series.unique().tolist() if c)

    # Which localities belong to which city, so the locality list narrows
    # down once a city is picked.
    localities_by_city = (
        data.dropna(subset=["City", "Locality"])
        .groupby("City")["Locality"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )

    currencies = CURRENCY_OPTIONS
    if "Currency" in data.columns:
        cleaned = (
            data["Currency"].dropna()
            .str.replace(r"\(.*?\)", "", regex=True)
            .str.strip()
        )
        found = sorted(c for c in cleaned.unique().tolist() if c)
        if found:
            currencies = found

    countries = list(COUNTRY_OPTIONS.keys())
    # Which cities belong to which country, so the city list narrows down
    # once a country is picked.
    cities_by_country: dict[str, list[str]] = {}
    if "Country Code" in data.columns:
        present_codes = data["Country Code"].dropna().unique().tolist()
        named = [COUNTRY_CODE_TO_NAME.get(c) for c in present_codes]
        named = sorted(n for n in named if n)
        if named:
            countries = named

        for country, group in data.dropna(subset=["Country Code", "City"]).groupby("Country Code"):
            country_label = COUNTRY_CODE_TO_NAME.get(country)
            if country_label:
                cities_by_country[country_label] = sorted(group["City"].unique().tolist())

    cities = sorted(data["City"].dropna().unique().tolist())

    return cities, cuisines, localities_by_city, currencies, countries, cities_by_country


if df is not None:
    (CITY_LIST, CUISINE_LIST, LOCALITIES_BY_CITY, CURRENCY_LIST,
     COUNTRY_LIST, CITIES_BY_COUNTRY) = get_dropdown_options(df)
else:
    CITY_LIST, CUISINE_LIST, LOCALITIES_BY_CITY, CURRENCY_LIST, COUNTRY_LIST = (
        [], [], {}, CURRENCY_OPTIONS, list(COUNTRY_OPTIONS.keys())
    )
    CITIES_BY_COUNTRY = {}

# ----------------------------------------------------------------------
# Sidebar - page picker
# ----------------------------------------------------------------------
st.sidebar.title("🍽️ Menu")
page = st.sidebar.radio(
    "Choose a page",
    ["📊 Restaurant Insights", "⭐ Rating Estimator"],
)

# ========================================================================
# PAGE 1 : RESTAURANT INSIGHTS  (Plotly charts)
# ========================================================================
if page == "📊 Restaurant Insights":
    st.markdown(
        """
        <div class="main-banner">
            <h1>📊 Restaurant Insights</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.error(
            "I couldn't find **Dataset.csv** next to this app. "
            "Please add the file to the same folder and reload the page."
        )
        st.stop()

    data = df.copy()

    # -------- top numbers --------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Restaurants", f"{len(data):,}")
    col2.metric("Cities Covered", f"{data['City'].nunique():,}")
    col3.metric("Average Rating", f"{data['Aggregate rating'].mean():.2f} / 5")
    col4.metric(
        "Offer Table Booking",
        f"{(data['Has Table booking'] == 'Yes').mean() * 100:.0f}%",
    )

    st.divider()

    # -------- rating distribution --------
    st.subheader("How are restaurants rated overall?")
    rated = data[data["Aggregate rating"] > 0]
    fig = px.histogram(
        rated, x="Aggregate rating", nbins=20,
        color_discrete_sequence=["#FF6B6B"],
        labels={"Aggregate rating": "Rating"},
    )
    fig.update_layout(bargap=0.05, yaxis_title="Number of restaurants")
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- rating category breakdown --------
    if "Rating text" in data.columns:
        st.subheader("How do ratings break down into categories?")
        rt_counts = (
            data[data["Aggregate rating"] > 0]["Rating text"]
            .value_counts().reset_index()
        )
        rt_counts.columns = ["Rating Category", "Count"]
        rt_color_map = {
            "Excellent": "#1DD3B0", "Very Good": "#4ECDC4",
            "Good": "#FFD93D", "Average": "#F4A261",
            "Poor": "#FF6B6B", "Not rated": "#B0B0B0",
        }
        fig = px.pie(
            rt_counts, names="Rating Category", values="Count", hole=0.45,
            color="Rating Category", color_discrete_map=rt_color_map,
        )
        fig.update_traces(textinfo="percent+label", pull=[0.03] * len(rt_counts))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- popular cuisines & cities --------
    left, right = st.columns(2)

    with left:
        st.subheader("Most common cuisines")
        cuisine_series = (
            data["Cuisines"].dropna().str.split(",").explode().str.strip()
        )
        top_cuisines = cuisine_series.value_counts().head(10).reset_index()
        top_cuisines.columns = ["Cuisine", "Count"]
        fig = px.bar(
            top_cuisines, x="Count", y="Cuisine", orientation="h",
            color="Count", color_continuous_scale="Plasma",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Cities with the most restaurants")
        top_cities = data["City"].value_counts().head(10).reset_index()
        top_cities.columns = ["City", "Count"]
        fig = px.bar(
            top_cities, x="Count", y="City", orientation="h",
            color="Count", color_continuous_scale="Teal",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- cuisine treemap --------
    st.subheader("A bird's-eye view of every cuisine on offer")
    cuisine_tree = cuisine_series.value_counts().head(30).reset_index()
    cuisine_tree.columns = ["Cuisine", "Count"]
    fig = px.treemap(
        cuisine_tree, path=["Cuisine"], values="Count",
        color="Count", color_continuous_scale="Turbo",
    )
    fig.update_traces(textinfo="label+value")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- top rated restaurants --------
    if "Restaurant Name" in data.columns:
        st.subheader("🏆 Top 10 highest-rated, most-voted restaurants")
        top_spots = (
            data[data["Votes"] > 100]
            .sort_values(["Aggregate rating", "Votes"], ascending=False)
            .head(10)
        )
        fig = px.bar(
            top_spots, x="Aggregate rating", y="Restaurant Name", orientation="h",
            color="Aggregate rating", color_continuous_scale="Sunsetdark",
            hover_data=["City", "Votes"] if "City" in top_spots.columns else ["Votes"],
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- price range & currency spread --------
    left, right = st.columns(2)
    with left:
        st.subheader("How restaurants are priced")
        price_labels = {1: "Low cost", 2: "Affordable", 3: "Mid-range", 4: "Expensive"}
        price_counts = data["Price range"].map(price_labels).value_counts().reset_index()
        price_counts.columns = ["Price Level", "Count"]
        fig = px.pie(
            price_counts, names="Price Level", values="Count", hole=0.45,
            category_orders={"Price Level": ["Low cost", "Affordable", "Mid-range", "Expensive"]},
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "Average Cost for two" in data.columns:
            st.subheader("Typical cost for two people")
            cost_data = data[data["Average Cost for two"] > 0]
            fig = px.histogram(
                cost_data, x="Average Cost for two", nbins=30,
                color_discrete_sequence=["#6C5CE7"],
            )
            fig.update_traces(marker_line_color="white", marker_line_width=1)
            fig.update_layout(yaxis_title="Number of restaurants")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- price range vs rating --------
    st.subheader("Does a pricier restaurant mean a better rating?")
    price_labels = {1: "Low cost", 2: "Affordable", 3: "Mid-range", 4: "Expensive"}
    price_df = data.copy()
    price_df["Price Level"] = price_df["Price range"].map(price_labels)
    fig = px.box(
        price_df, x="Price Level", y="Aggregate rating",
        category_orders={"Price Level": ["Low cost", "Affordable", "Mid-range", "Expensive"]},
        color="Price Level",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- table booking / online delivery --------
    st.subheader("Table booking & online delivery")
    left, right = st.columns(2)
    with left:
        counts = data["Has Table booking"].value_counts().reset_index()
        counts.columns = ["Offers Table Booking", "Count"]
        fig = px.pie(
            counts, names="Offers Table Booking", values="Count", hole=0.5,
            color_discrete_sequence=["#4ECDC4", "#FF6B6B"],
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        counts = data["Has Online delivery"].value_counts().reset_index()
        counts.columns = ["Offers Online Delivery", "Count"]
        fig = px.pie(
            counts, names="Offers Online Delivery", values="Count", hole=0.5,
            color_discrete_sequence=["#FFD93D", "#6C5CE7"],
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- which cuisines get the best ratings --------
    st.subheader("Which cuisines tend to get the best ratings?")
    cuisine_rating = data[["Cuisines", "Aggregate rating"]].dropna().copy()
    cuisine_rating["Cuisines"] = cuisine_rating["Cuisines"].str.split(",")
    cuisine_rating = cuisine_rating.explode("Cuisines")
    cuisine_rating["Cuisines"] = cuisine_rating["Cuisines"].str.strip()
    top_by_count = cuisine_rating["Cuisines"].value_counts().head(15).index
    avg_by_cuisine = (
        cuisine_rating[cuisine_rating["Cuisines"].isin(top_by_count)]
        .groupby("Cuisines")["Aggregate rating"].mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.bar(
        avg_by_cuisine, x="Aggregate rating", y="Cuisines", orientation="h",
        color="Aggregate rating", color_continuous_scale="Sunsetdark",
        labels={"Aggregate rating": "Average Rating"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- table booking / delivery effect on rating --------
    st.subheader("Do table booking & online delivery lead to better ratings?")
    left, right = st.columns(2)
    with left:
        avg_tb = data.groupby("Has Table booking")["Aggregate rating"].mean().reset_index()
        fig = px.bar(
            avg_tb, x="Has Table booking", y="Aggregate rating", color="Has Table booking",
            labels={"Aggregate rating": "Average Rating"}, text_auto=".2f",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        avg_od = data.groupby("Has Online delivery")["Aggregate rating"].mean().reset_index()
        fig = px.bar(
            avg_od, x="Has Online delivery", y="Aggregate rating", color="Has Online delivery",
            labels={"Aggregate rating": "Average Rating"}, text_auto=".2f",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- votes vs rating --------
    st.subheader("Do more popular restaurants get higher ratings?")
    scatter_data = data[data["Aggregate rating"] > 0].copy()
    price_labels = {1: "Low cost", 2: "Affordable", 3: "Mid-range", 4: "Expensive"}
    scatter_data["Price Level"] = scatter_data["Price range"].map(price_labels)
    fig = px.scatter(
        scatter_data, x="Votes", y="Aggregate rating", color="Price Level",
        opacity=0.6, log_x=True,
        labels={"Votes": "Number of customer votes (log scale)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- top countries --------
    if "Country Code" in data.columns:
        st.subheader("Which countries have the most restaurants here?")
        country_counts = data["Country Code"].value_counts().reset_index()
        country_counts.columns = ["Country Code", "Count"]
        country_counts["Country"] = country_counts["Country Code"].map(COUNTRY_CODE_TO_NAME)
        country_counts["Country"] = country_counts["Country"].fillna(
            country_counts["Country Code"].astype(str)
        )
        country_counts = country_counts.sort_values("Count", ascending=False).head(10)
        fig = px.bar(
            country_counts, x="Count", y="Country", orientation="h",
            color="Count", color_continuous_scale="Purp",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------- map --------
    if "Latitude" in data.columns and "Longitude" in data.columns:
        st.subheader("Where are these restaurants located?")
        map_data = data.dropna(subset=["Latitude", "Longitude"]).sample(
            min(2000, len(data)), random_state=1
        )
        fig = px.scatter_mapbox(
            map_data, lat="Latitude", lon="Longitude",
            hover_name="Restaurant Name" if "Restaurant Name" in map_data.columns else None,
            color="Aggregate rating", color_continuous_scale="RdYlGn",
            zoom=1, height=500,
        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# PAGE 2 : RATING ESTIMATOR  (uses the saved model)
# ========================================================================
else:
    st.markdown(
        """
        <div class="main-banner">
            <h1>⭐ Rating Estimator</h1>
            <p>Tell us a bit about the restaurant, and we'll estimate the rating customers are likely to give it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "I couldn't find **best_model.pkl** next to this app. "
            "Please add the saved model file to the same folder and reload the page."
        )
        st.stop()

    # ---- pickers that live OUTSIDE the form, so the city list narrows down
    # by country, and the locality list narrows down by city ----
    st.subheader("Basic details")
    c1, c2 = st.columns(2)
    with c1:
        country_name = st.selectbox("Country", COUNTRY_LIST)

        cities_for_country = CITIES_BY_COUNTRY.get(country_name, CITY_LIST)
        city_choices = cities_for_country + [OTHER_LABEL]
        city_pick = st.selectbox("City", city_choices if city_choices else [OTHER_LABEL])
        if city_pick == OTHER_LABEL:
            city = st.text_input("Type the city name", placeholder="e.g. New Delhi")
        else:
            city = city_pick

        locality_options = LOCALITIES_BY_CITY.get(city, []) + [OTHER_LABEL]
        locality_pick = st.selectbox("Area / Locality", locality_options)
        if locality_pick == OTHER_LABEL:
            locality = st.text_input("Type the area / locality", placeholder="e.g. Connaught Place")
        else:
            locality = locality_pick

    with c2:
        currency = st.selectbox("Currency used for pricing", CURRENCY_LIST)

        cuisine_pick = st.multiselect(
            "Cuisines served (pick one or more)",
            CUISINE_LIST,
            placeholder="e.g. North Indian, Chinese",
        )
        extra_cuisine = st.text_input(
            "Not in the list? Add more cuisines here (comma separated)",
            placeholder="optional",
        )
        cuisines = ", ".join([c for c in cuisine_pick] + (
            [extra_cuisine] if extra_cuisine else []
        ))

    with st.form("rating_form"):
        st.subheader("Pricing & popularity")
        c3, c4, c5 = st.columns(3)
        with c3:
            cost_for_two = st.number_input(
                "Average cost for two people", min_value=0, value=500, step=50
            )
        with c4:
            price_level = st.select_slider(
                "How pricey is it?",
                options=[1, 2, 3, 4],
                format_func=lambda x: {1: "Low cost", 2: "Affordable",
                                        3: "Mid-range", 4: "Expensive"}[x],
                value=2,
            )
        with c5:
            votes = st.number_input(
                "Number of customer votes/reviews so far", min_value=0, value=50
            )

        st.subheader("Services offered")
        c6, c7 = st.columns(2)
        with c6:
            table_booking = st.radio("Table booking available?", ["Yes", "No"], horizontal=True)
        with c7:
            online_delivery = st.radio("Online delivery available?", ["Yes", "No"], horizontal=True)

        with st.expander("Optional: exact map location"):
            longitude = st.number_input(
                "Longitude (from Google Maps, if you have it)",
                value=0.0, format="%.6f",
            )

        submitted = st.form_submit_button("Estimate the rating ⭐", use_container_width=True)

    if submitted:
        input_row = pd.DataFrame([{
            "Longitude": longitude,
            "Average Cost for two": cost_for_two,
            "Country Code": COUNTRY_OPTIONS[country_name],
            "Price range": price_level,
            "Votes": votes,
            "City": city if city else "Unknown",
            "Currency": currency,
            "Has Table booking": table_booking,
            "Has Online delivery": online_delivery,
            "Locality": locality if locality else "",
            "Cuisines": cuisines if cuisines else "",
        }])

        try:
            predicted = model.predict(input_row)[0]
            predicted = float(np.clip(predicted, 0, 5))
        except Exception as e:
            st.error(f"Something went wrong while estimating the rating: {e}")
            st.stop()

        if predicted >= 4.5:
            verdict, color = "Excellent 🤩", "#1B5E20"
        elif predicted >= 4.0:
            verdict, color = "Very Good 😃", "#2E7D32"
        elif predicted >= 3.5:
            verdict, color = "Good 🙂", "#F9A825"
        elif predicted >= 2.5:
            verdict, color = "Average 😐", "#EF6C00"
        else:
            verdict, color = "Needs Improvement 😕", "#C62828"

        st.divider()
        st.subheader("Here's what we think")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Estimated Rating", f"{predicted:.1f} / 5")
        with c2:
            st.markdown(
                f"<h3 style='color:{color};'>{verdict}</h3>",
                unsafe_allow_html=True,
            )
        st.caption(
            "This is just an estimate based on similar restaurants in the data. "
            "The real rating can differ once actual customers start reviewing."
        )