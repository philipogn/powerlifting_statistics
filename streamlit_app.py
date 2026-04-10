import streamlit as st

def configure_page():
    st.set_page_config(page_title="Powerlifting Total Predictor", layout="wide")
    st.title("Powerlifting Predictor")
    st.write("Hello World")
    if st.button("clickme"):
        st.write("Hello World")

    st.sidebar.success("test")

'''
username input
age, bodyweight input
predicted, current, improvement in kgs
list comp history?
'''


def main():
    configure_page()

if __name__ == "__main__":
    main()