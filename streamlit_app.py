import streamlit as st
from pinecone_search import pinecone_search

# Define the Streamlit app
def main():

    st.title("Image Search")

    tab1 = st.tabs(["Pinecone search"])

    with tab1:
        pinecone_search()

if __name__ == "__main__":
    main()