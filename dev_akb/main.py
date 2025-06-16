import streamlit as st

def main():
    st.title("Crimmetrics Dashboard")
    st.write("Welcome to the Crimmetrics basic Streamlit UI!")

    name = st.text_input("Enter your name:")
    if name:
        st.success(f"Hello, {name}!")

    number = st.slider("Pick a number", 1, 100, 50)
    st.write(f"You selected: {number}")

    if st.button("Click Me"):
        st.info("Button clicked!")

if __name__ == "__main__":
    main()