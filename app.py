import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic CSV Bulk Predictor")

# 1. File Uploader for CSV
uploaded_file = st.file_uploader("Upload your passenger CSV file", type=['csv'])

if uploaded_file is not None:
    # Read the CSV file
    input_df = pd.read_csv(uploaded_file)

    st.write("### Raw Data Preview")
    st.dataframe(input_df.head())

    try:
        # 2. Preprocessing
        # We must transform the raw CSV data into the 16 features the model expects
        input_df=input_df.drop(["PassengerId","Name","Ticket"],axis=1)

        median_age = input_df['Age'].median()
        median_fare=input_df["Fare"].median()
        input_df["Age"]=input_df["Age"].fillna(median_age)
        input_df["Fare"]=input_df["Fare"].fillna(median_fare)

        input_df['Deck'] = input_df['Cabin'].str[0]
        input_df['Deck']=input_df['Deck'].fillna('U')

        input_df['Family count']=input_df["SibSp"]+input_df["Parch"]+1

        input_df=input_df.drop(["Cabin","SibSp","Parch"],axis=1)

        processed_df = pd.get_dummies(input_df, columns=["Pclass","Sex","Deck","Embarked"],drop_first=True, dtype=int)


        # List of the 16 columns your model was trained on (update this list to match your exact X_train)
        model_columns = [
            'Age', 'Fare', 'Family count', 'Pclass_2', 'Pclass_3', 'Sex_male',
            'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G',
            'Deck_T', 'Deck_U', 'Embarked_Q', 'Embarked_S'
        ]

        # This one line handles both missing columns AND extra columns automatically
        processed_df = processed_df.reindex(columns=model_columns, fill_value=0)

        # 3. Predict
        if st.button("Generate Predictions"):
            predictions = model.predict(processed_df)
            probabilities = model.predict_proba(processed_df)[:, 1]

            # Attach results to the display dataframe
            output_df = input_df.copy()
            output_df['Survived_Pred'] = predictions
            output_df['Survival_Prob'] = probabilities

            st.write("### Results")
            st.dataframe(output_df)

            # 4. Export as CSV
            csv_data = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Resulting CSV",
                data=csv_data,
                file_name="titanic_results.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Processing Error: {e}")