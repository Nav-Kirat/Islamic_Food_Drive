import streamlit as st

# Set page title and layout
st.set_page_config(page_title="About Us", layout="centered")

# About Us Page Title
st.title("🤝 About Us")



# Team Member Details
team_members = [{
        "name": "Dionathan Dos Santos",
        "linkedin": "https://www.linkedin.com/in/dionathanadiel",
        "role": "Model Developer",
    },
    {
        "name": "Navkirat singh",
        "linkedin": "https://linkedin.com/in/navkirat",
        "role": "Team Lead",
    },
    {
        "name": "Mayank khera",
        "linkedin": "https://www.linkedin.com/in/mayank-khera-915b12252/",
        "role": "Data gathering and preparation (Scrum Master)",
    },
    {
        "name": "Vinit Kataria",
        "linkedin": "https://www.linkedin.com/in/vinit-kataria-46b13b222/",
        "role": "Insights Analyst",
    },
    {
        "name": "Harshdeep Kaur",
        "linkedin": "https://www.linkedin.com/in/harshdeep-kaur-714b62118/",
        "role": "Research Coordinator",
    },
    
]

# Create Columns for Team Members
cols = st.columns(5)

# Add each team member to a column
for col, member in zip(cols, team_members):
    with col:
        st.markdown(
            f"""
            ### [{member['name']}]({member['linkedin']})
            """,
            unsafe_allow_html=True,
        )
        st.write(f"**Role**: {member['role']}")
