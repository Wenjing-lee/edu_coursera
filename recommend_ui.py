from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
import json

def load_courses(filename="coursera_results.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ JSON file not found.")
        return []

def filter_courses(courses, keyword="", partner="", limit=5):
    results = []

    for course in courses:
        title = course["title"].lower()
        partner_name = course["partner"].lower()

        if keyword.lower() not in title:
            continue
        if partner and partner.lower() not in partner_name:
            continue

        results.append(course)

    return results[:limit or None]

# App UI
st.title("🎓 Coursera Course Recommender")

courses = load_courses()
titles = [course["title"] for course in courses]

tab1, tab2 = st.tabs(["🔍 Filter Courses", "Recommend Similar"])

# TAB 1: Filtering
with tab1:
    keyword = st.text_input("🔍 Enter keyword (e.g. python, data, AI)")
    partner = st.text_input("🏫 Filter by partner (e.g. Google, IBM, Michigan)")
    limit = st.slider("🔢 Max results to show", min_value=1, max_value=20, value=5)

    if st.button("Show Recommendations"):
        matches = filter_courses(courses, keyword, partner, limit)

        st.success(f"Found {len(matches)} course(s)")

        for course in matches:
            st.subheader(course["title"])
            st.write(f"**Partner:** {course['partner']}")
            st.write(f"[🔗 View Course]({course['url']})")
            st.markdown("---")

# TAB 2: Similar Course Recommender
with tab2:
    st.subheader("Find Similar Courses")
    selected = st.selectbox("Pick a course you like:", titles)

    def get_recommendations(selected_title, top_n=5):
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(titles)

        try:
            index = titles.index(selected_title)
        except ValueError:
            return []

        cosine_sim = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
        similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
        return [courses[i] for i in similar_indices]

    if st.button("Recommend Similar Courses"):
        results = get_recommendations(selected)

        if not results:
            st.error("No recommendations found.")
        else:
            st.success(f"Here are courses similar to: *{selected}*")
            for course in results:
                st.markdown(f"**{course['title']}** ({course['partner']})")
                st.markdown(f"[🔗 View Course]({course['url']})")
                st.markdown("---")
