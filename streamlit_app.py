import streamlit as st
from pinecone_search import pinecone_search

# Define the Streamlit app
def main():
    st.title("Fashion Image Search")
    pinecone_search()

if __name__ == "__main__":
    main()
