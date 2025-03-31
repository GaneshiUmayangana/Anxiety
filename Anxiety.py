import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page Configurations
st.set_page_config(page_title="Student Anxiety Prediction", layout="wide")


col1, col2,col3 = st.columns([1,1,1])
    
if "page" not in st.session_state:
    st.session_state.page = "Home"
    
with col1:
    if st.button("🏠 Home", key="home_btn"):
        st.session_state.page = "Home"
with col2:
    if st.button("📖 Methodology", key="home_btn"):
        st.session_state.page = "Methodology"        
with col3:
    if st.button("📊 Prediction", key="prediction_btn"):
        st.session_state.page = "Prediction"


if st.session_state.page == "Home":
    st.markdown("""
    <style>
        .header {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: #F5E8C7;
            background-color: #3E2723;
            padding: 20px;
            border-radius: 10px;
        }
        .subheader {
            text-align: center;
            font-size: 25px;
            color: #BCAAA4;
        }
        .content {
            font-size: 18px;
            text-align: justify;
            color: white;
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .stButton>button {
            font-size: 20px;
            font-weight: bold;
            width: 200px;
            color: white;
            background-color: #795548;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Display Header
st.markdown("<div class='header'>Student Anxiety Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Understand and manage student anxiety effectively</div>", unsafe_allow_html=True)



if st.session_state.page == "Methodology":
    st.markdown("""
        <style>
            .stApp {
                background-image: url('https://img.freepik.com/free-vector/vintage-woman-flowers-outline_53876-99109.jpg');
                background-size: cover;
                background-position: center;
            }    
            .header {
                text-align: center;
                font-size: 50px;  
                font-weight: bold;
                color: black;
                padding: 10px;
            }
            .subheader {
                text-align: center;
                font-size: 30px;  
                color: black;
            }
        </style>
        <div class="header">
            Student Anxiety Prediction
        </div>
        <div class="subheader">
            Learn more about adolescent anxiety and its impact
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <h2 style="color: black;">What is Anxiety?</h2>
    <p style="text-align: justify; font-size: 18px;">
        Anxiety is a complex emotional state characterized by feelings of fear, 
        dread, and uneasiness, often accompanied by physical symptoms like 
        restlessness, sweating, and a rapid heartbeat. Unlike fear, which is triggered 
        by a specific threat, anxiety is a broader emotional response influenced by various 
        cognitive and affective processes. Researchers suggest that anxiety is not just a singular 
        emotion but a blend of different feelings and thoughts, making it difficult to define. Anxiety disorders, 
        such as social anxiety, panic disorder, and generalized anxiety disorder, can arise due to multiple risk 
        factors, including low self-esteem, childhood trauma, family history of depression, and a challenging social
        environment. These disorders not only impact mental well-being but are also linked to higher rates of medical 
        conditions, highlighting the importance of early identification and intervention.
    </p>
    
    <h2 style="color: black;">Importance of Identifying Adolescent Anxiety</h2>
    <p style="text-align: justify; font-size: 18px;">
        Identifying anxiety in adolescents is crucial for their health, happiness, and academic success. 
        This study helps recognize anxiety-related factors, including demographic details, family & social support, 
        health & well-being, and academic & social support. 
        Early intervention allows educators, parents, and mental health professionals to provide targeted support, 
        improving emotional well-being and academic performance.
    </p>
    
    <p style="text-align: justify; font-size: 18px;">
        The rise in adolescent psychological health issues like anxiety highlights the challenges of puberty, 
        including spiritual, physical, psychological, and cognitive changes. Early identification can help mitigate 
        these struggles, ensuring a healthier and more stable transition into adulthood.
    </p>
    <hr>
""", unsafe_allow_html=True)
    # Embed YouTube video after the paragraph
    st.markdown("""
        <iframe width="560" height="315" src="https://www.youtube.com/embed/wr4N-SdekqY" frameborder="0" 
        allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    """, unsafe_allow_html=True)

if st.session_state.page == "Prediction":
    st.markdown("""
    <style>
        .stApp {
            background-image: url('https://img.freepik.com/free-vector/vintage-woman-flowers-outline_53876-99109.jpg');
            background-size: cover;
            background-position: center;
        }    
        .header {
            text-align: center;
            font-size: 50px;  /* Increased font size */
            font-weight: bold;
            color: black;
            padding: 10px;
        }
        .subheader {
            text-align: center;
            font-size: 30px;  /* Increased font size */
            color: black;
        }
        .stSelectbox, .stNumberInput, .stTextInput, .stTextArea {
            font-size: 25px;  /* Increased input font size */
            width: 100%;  /* Widen the input boxes */
        }
        .stButton>button {
            font-size: 30px;  /* Increased button text size */
            font-weight: bold;  /* Make button text bold */
            width: 100%;  /* Stretch the button to full width */
            color: white;  /* Button text color */
            background-color: #000080;  /* Button background color */
        }
        .stRadio, .stCheckbox {
            font-size: 25px;  /* Increased radio/checkbox font size */
        }
        .stMarkdown h3 {
            font-size: 30px;  /* Increased header font size for results */
        }
    </style>
    <div class="header">
        Student Anxiety Prediction
    </div>
    <div class="subheader">
        Predict Anxiety Levels Based on Various Factors
    </div>
""", unsafe_allow_html=True)



    # **Step 1: Load the trained model & encoders**
    with open("best_rf.pkl", "rb") as f:
        model = pickle.load(f)

    with open("ordinal_encoder.pkl", "rb") as f:
        ordinal_encoder = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        one_hot_encoder = pickle.load(f)

    with open('multi_target_classifier(1).pkl', 'rb') as model_file:
        multi_model = pickle.load(model_file)


# **Step 2: Define input fields**
#st.title("Student Anxiety Prediction")

# **User Inputs**
# Create two columns
    col1, col2 = st.columns(2)

    with col1:
        school_type = st.selectbox("School Type", ["Below 500", "501-1500", "Above 1500"])
        age = st.number_input("Age", min_value=14, max_value=18)
        religion = st.selectbox("Religion", ['Buddhism', 'Christianity / Catholicism', 'Islam', 'Hinduism'])
        distance = st.selectbox("Distance to School", ["Less than 2", "2 -5", "More than 5"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, step=0.1)
        current_living_status = st.selectbox("Current Living Status", ['Living with one parent', 'Living with both parents', 'Living with guardians', 'Other'])
        parental_employment_status = st.selectbox("Parental Employment", ['Only one of my parents is employed', 'Both my parents are employed', 'Neither of my parents are employed'])
        siblings = st.selectbox("Do you have Siblings", ["No", "Yes"])
        screen_hours = st.selectbox("Daily Screen Time", ["Less than 2 hours", "2-4 hours", "4-6 hours", "More than 6 hours"])
        academic_performance = st.selectbox("Rate Your Academic Performance", ['Below average', 'Average', 'Above average'])
        ask_teacher_help = st.selectbox("Comfortable Asking Teachers for Help", ['Never ask', 'Not comfortable', 'Somewhat comfortable','Very comfortable'])

    with col2:
        grade = st.selectbox("Grade", ["Grade 10", "Grade 11"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        support = st.selectbox("Feel Supported by Family", ["Never", "Rarely", "Sometimes", "Always"])
        transportation = st.selectbox("Transportation Mode", ['Motor bike', 'Bus', 'School Van', 'Car', 'Other'])
        sleep_hours = st.selectbox("Sleeping Hours", ["Less than 5 hours", "5-7 hours", "7-9 hours", "More than 9 hours"])
        discuss_worries = st.selectbox("Discuss Worries with Family/Friends", ["Never", "Rarely", "Sometimes", "Always"])
        income = st.selectbox("Family Income", [
            "Less than 30,000", "30,000 – 100,000", "100,000 – 250,000", "250,000 – 500,000", "Greater than 500,000"
        ])
        physical_activity = st.selectbox("Physical Activity", ["Never", "Rarely", "Yes, a few times a week", "Yes, daily"])
        screen_affect_sleep = st.selectbox("Screen Time Affects Sleep", ["No", "Maybe a little", "Yes, definitely"])
        stress = st.selectbox("Stress Due to Schoolwork/Exams", ["Never", "Rarely", "Sometimes", "Always"])
        enjoy_school = st.selectbox("Do You Enjoy School?", ['No', 'Sometimes', 'Yes'])


# **Step 3: Convert inputs into DataFrame**
    features = pd.DataFrame([[school_type, age, grade, gender, religion, distance, transportation, siblings,
                          current_living_status, parental_employment_status, income, support, discuss_worries, bmi,
                          sleep_hours, physical_activity, screen_hours, screen_affect_sleep, academic_performance,
                          stress, ask_teacher_help, enjoy_school]],
                        columns=['School_Type', 'Age', 'Grade', 'Gender', 'Religion', 'Distance',
                                 'Transportation', 'Siblings', 'Current_living_status',
                                 'Parental_employment_status', 'Total_monthly_family_income_rupees',
                                 'Feel_about_supported_of_family',
                                 'discuss_worries_or_problems_with_family_or_friends', 'BMI',
                                 'Sleeping_Hours', 'Engage_in_physical_activity', 'Screen_Hours',
                                 'Screen_time_affect_sleep', 'Given_rate_of_academic_performance',
                                 'Stressed_due_to_schoolwork_or_exams',
                                 'Comfortable_of_asking_teachers_for_help',
                                 'Enjoyability_of_going_to_school'])

# **Step 4: Apply Encoding**
    ordinal_columns = ['Grade', 'Distance', 'Total_monthly_family_income_rupees', 'Sleeping_Hours', 'Screen_Hours',
                    'Feel_about_supported_of_family', 'Engage_in_physical_activity', 'Screen_time_affect_sleep', 'Stressed_due_to_schoolwork_or_exams']

    one_hot_columns = ['School_Type', 'Gender', 'Religion', 'Transportation', 'Siblings',
                    'Current_living_status', 'Parental_employment_status',
                    'discuss_worries_or_problems_with_family_or_friends',
                    'Given_rate_of_academic_performance',
                    'Comfortable_of_asking_teachers_for_help',
                    'Enjoyability_of_going_to_school']

# Apply Ordinal Encoding
    features[ordinal_columns] = ordinal_encoder.transform(features[ordinal_columns])

# Apply One-Hot Encoding
    encoded_one_hot = one_hot_encoder.transform(features[one_hot_columns])
    encoded_one_hot_df = pd.DataFrame(encoded_one_hot, columns=one_hot_encoder.get_feature_names_out(one_hot_columns))

# Drop original categorical columns and combine ordinal + one-hot encoded features
    features_numeric = features.drop(columns=one_hot_columns + ordinal_columns)
    features_encoded = pd.concat([features_numeric, features[ordinal_columns], encoded_one_hot_df], axis=1)

# Ensure all expected columns from training exist in test input
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in features_encoded.columns:
            features_encoded[col] = 0  # Add missing columns with zeros

# Reorder columns to match the training data
    features_encoded = features_encoded[expected_columns]



# Step 6: Reset previous predictions on input change
    def reset_predictions():
        st.session_state.anxiety_prediction = None
        st.session_state.anxiety_prediction_proba = None
        st.session_state.sub_anxiety_prediction = None
        st.session_state.sub_anxiety_prediction_proba = None

# Reset predictions when any of the inputs change
    inputs = [school_type, age, grade, gender, religion, distance, transportation, siblings, current_living_status,
            parental_employment_status, income, support, discuss_worries, bmi, sleep_hours, physical_activity,
            screen_hours, screen_affect_sleep, academic_performance, stress, ask_teacher_help, enjoy_school]

    if any(input != getattr(st.session_state, f"last_{i}", None) for i, input in enumerate(inputs)):
        reset_predictions()

# Store current inputs to detect changes
    for i, input in enumerate(inputs):
        st.session_state[f"last_{i}"] = input

    if st.button("Predict Anxiety Status"):
        prediction = model.predict(features_encoded)[0]  # Single prediction
        prediction_proba = model.predict_proba(features_encoded)[0]  # Probability distribution

    # Interpret results
        anxiety_labels = {0: "No Anxiety", 1: "Possible Anxiety Disorder", 2: "Specific Anxiety Disorder"}
        predicted_label = anxiety_labels[prediction]

    # Store results in session state
        st.session_state.anxiety_prediction = predicted_label
        st.session_state.anxiety_prediction_proba = prediction_proba

# Step 7: Make Prediction for Sub Anxiety Status using the multi_model
    if 'sub_anxiety_prediction' not in st.session_state:
        st.session_state.sub_anxiety_prediction = None
    if 'sub_anxiety_prediction_proba' not in st.session_state:
        st.session_state.sub_anxiety_prediction_proba = None

# Display Anxiety Prediction Results (if anxiety results exist)
    if st.session_state.anxiety_prediction is not None:
        st.markdown(f"<h3 style='color: blue; font-weight: bold;'>Predicted Anxiety Level: </h3><h3 style='color: red; font-weight: bold;'>{st.session_state.anxiety_prediction}</h3>", unsafe_allow_html=True)
        st.write("#### Prediction Probabilities:")
    
        anxiety_labels = {0: "No Anxiety", 1: "Possible Anxiety Disorder", 2: "Specific Anxiety Disorder"}
        for i, label in anxiety_labels.items():
            st.write(f"{label}: {st.session_state.anxiety_prediction_proba[i]:.2f}")



    if st.button("Predict Sub Anxiety Status"):
    # Make predictions with the multi-target classifier model
        multi_prediction = multi_model.predict(np.array(features_encoded))[0]  # The first set of predictions (for one student)
        multi_prediction_proba = multi_model.predict_proba(np.array(features_encoded) ) # List of probability distributions

    # Define the sub-anxiety labels
        sub_anxiety_labels = {
            0: "Panic Disorder",
            1: "Generalized Anxiety Disorder",
            2: "Separation Anxiety Disorder",
            3: "Social Anxiety Disorder",
            4: "School Avoidance"
        }

    # Store results in session state
        st.session_state.sub_anxiety_prediction = multi_prediction
        st.session_state.sub_anxiety_prediction_proba = multi_prediction_proba

# Display Sub Anxiety Prediction Results (if sub-anxiety results exist)
    if st.session_state.sub_anxiety_prediction is not None:
        st.markdown("<h3 style='color: blue; font-weight: bold;'>Predicted Sub Anxiety Status:</h3>", unsafe_allow_html=True)

    # Loop through each sub-anxiety category and its corresponding probability
        sub_anxiety_labels = {
            0: "Panic Disorder",
            1: "Generalized Anxiety Disorder",
            2: "Separation Anxiety Disorder",
            3: "Social Anxiety Disorder",
            4: "School Avoidance"
        }

        for i, label in sub_anxiety_labels.items():
            if isinstance(st.session_state.sub_anxiety_prediction_proba[i], np.ndarray):
                probability = float(st.session_state.sub_anxiety_prediction_proba[i][0][1])  # Extract the probability for the positive class
            else:
             probability = float(st.session_state.sub_anxiety_prediction_proba[i])  # Convert directly if it's already a scalar
        
       

    # Display the final predicted sub-anxiety categories
        predicted_labels = [sub_anxiety_labels[i] for i in range(len(st.session_state.sub_anxiety_prediction)) if st.session_state.sub_anxiety_prediction[i] == 1]
    
        if predicted_labels:
            predicted_text = ", ".join(predicted_labels)
            st.markdown(f"<h4 style='color: red; font-weight: bold;'>{predicted_text}</h4>", unsafe_allow_html=True)
        else:
            st.write("No specific sub-anxiety disorder predicted.")




