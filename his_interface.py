import pandas as pd
import streamlit as st
import PyPDF2
from streamlit import session_state as ss
import psycopg2
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics

def getdata():
    # gather data sources from sql database
    conn_string = 'postgresql://postgres:postgres@localhost/postgres'
  
    db = create_engine(conn_string) 
    conn = db.connect() 
    conn1 = psycopg2.connect( 
        database="postgres", 
    user='postgres',  
    password='postgres',  
    host='localhost',  
    port= '5432'
    ) 

    conn1.autocommit = True
    cursor = conn1.cursor() 

    year0 = pd.read_sql_query('SELECT * FROM year0', con=db)

    year5 = pd.read_sql_query('SELECT * FROM year5', con=db)

    year10 = pd.read_sql_query('SELECT * FROM year10', con=db)

    conn.commit()
    conn.close()

    # merge the data sources
    temp_df = pd.merge(year0, year5, on='SWANID')
    allyears_df = pd.merge(temp_df, year10, on='SWANID')

    # clean data
    allyears_df = allyears_df.drop('index_x', axis=1)
    allyears_df = allyears_df.drop('index_y', axis=1)
    for column in allyears_df.columns:
        if column == "SWANID":
            continue
        allyears_df[column].replace([' '],['-1'],inplace=True)

    return allyears_df

def getInputs(question1, question2, question3, question4,
              question5, question6, question7, question8,
              question9, question10, question11):
        
        # convert questions from string to int to feed into our model
        # question 1
        if question1 == "Black or African American":
            q1 = 1
        elif question1 == "Asian":
            q1 = 2
        elif question1 == "White":
            q1 = 4
        else:
            q1 = 2
        
        # question 2
        if question2 == "Yes":
            q2 = 2
        else:
            q2 = 1
        
        # question 3
        if question3 == "Every day":
            q3 = 5
        elif question3 == "9-13 days":
            q3 = 4
        elif question3 == "6-8 days":
            q3 = 3
        elif question3 == "1-5 days":
            q3 = 2
        else:
            q3 = 1
        
        # question 4
        if question4 == "Every day":
            q4 = 5
        elif question4 == "9-13 days":
            q4 = 4
        elif question4 == "6-8 days":
            q4 = 3
        elif question4 == "1-5 days":
            q4 = 2
        else:
            q4 = 1
        
        # question 5
        if question5 == "Always":
            q5 = 1
        elif question5 == "Almost Always":
            q5 = 2
        elif question5 == "Sometimes":
            q5 = 3
        elif question5 == "Almost Never":
            q5 = 4
        elif question5 == "Never":
            q5 = 5
        else:
            q5 = 6

        # question 6
        if question6 == "Every day":
            q6 = 5
        elif question6 == "9-13 days":
            q6 = 4
        elif question6 == "6-8 days":
            q6 = 3
        elif question6 == "1-5 days":
            q6 = 2
        else:
            q6 = 1

        # question 7
        if question7 == "Live Birth":
            q7 = 1
        elif question7 == "Still Birth":
            q7 = 2
        elif question7 == "Miscarriage":
            q7 = 3
        elif question7 == "Abortion":
            q7 = 4
        elif question7 == "Tubal(Ectopic) Pregnancy":
            q7 = 5
        elif question7 == "Still Prgenant":
            q7 = 6
        else:
            q7 = -1

        # question 8
        if question8 == "Live Birth":
            q8 = 1
        elif question8 == "Still Birth":
            q8 = 2
        elif question8 == "Miscarriage":
            q8 = 3
        elif question8 == "Abortion":
            q8 = 4
        elif question8 == "Tubal(Ectopic) Pregnancy":
            q8 = 5
        elif question8 == "Still Prgenant":
            q8 = 6
        else:
            q8 = -1

        # question 9
        if question9 == "Live Birth":
            q9 = 1
        elif question9 == "Still Birth":
            q9 = 2
        elif question9 == "Miscarriage":
            q9 = 3
        elif question9 == "Abortion":
            q9 = 4
        elif question9 == "Tubal(Ectopic) Pregnancy":
            q9 = 5
        elif question9 == "Still Prgenant":
            q9 = 6
        else:
            q9 = -1
        
        # question 10
        if question10 == "Yes":
            q10 = 2
        else:
            q10 = 1
        
        # question 11
        if question11 == "Hysterectomy/both ovaries removed":
            q11 = 1
        elif question11 == "Post-menopausal":
            q11 = 2
        elif question11 == "Late perimenopause":
            q11 = 3
        elif question11 == "Early perimenopause":
            q11 = 4
        elif question11 == "Pre-menopausal":
            q11 = 5
        elif question11 == "Pregnant/breastfeeding":
            q11 = 6
        elif question11 == "Unknown due to hormones (HT) use":
            q11 = 7
        else:
            q11 = 8
        
        return q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11

# create the target column
# 1 means hormone therapy, and 0 means non-hormonal tretament
def conditions(df):
    #if ((df['SOYYSNO10'] == 2) or (df['ACUPMEN10'] == 2) or (df['BCOHMEN10'] == 2) or
    #    (df['DQUAMEN10'] == 2) or (df['DIETMEN10'] == 2) or (df['EXERMEN10'] == 2) or 
    #    (df['FLAXMEN10'] == 2) or (df['GINSMEN10'] == 2) or (df['GLUSMEN10'] == 2)):
    #    return 2
    if ((df['HORMPIL0'] == 2) or (df['ESTRPTC0'] == 2) or (df['COMBEVE0'] == 2) or (df['PTCHEVE0'] == 2) 
            or (df['ESTROG15'] == 2) or (df['ESTROG210'] == 2) or (df['ESTROG110'] == 2) or
            (df['PROGES110'] == 2) or (df['PROGES210'] == 2) or (df['PROGDA110'] == 2) 
            or (df['PROGDA210'] == 2) or (df['OHRM_110'] == 2) or (df['OHRM_210'] == 2) or
            (df['OHRM_310'] == 2) or (df['OHRM_410'] == 2)):
        return 1
    else:
        return 0

def createData(df):
    # convert columns to numeric
    new_df = df.apply(pd.to_numeric)
    new_df["Target"] = new_df.apply(conditions, axis=1)
    return new_df

def getModel(df):
    X = df[['RACE', 'PELVIC0',
       'MOODCHG0', 'OUTCM20', 'OUTCM30', "OUTCM10",
       'PROGES15', 'ESTLSTV5',
       'STATUS10',
       'HOTFLAS10',
       'VAGINDR10']]
    y = df.Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    return clf

def main(df, model):
    st.title("Menopausal Symptom Management CDSS")

    choice = st.sidebar.selectbox(
        "Select a page",
        options = ("Search for Patient", "Treatment Options")
    )


    if choice == "Search for Patient":
        st.write("Please enter a SwanID to find information regarding a specific patient.")
        swanID = st.text_input("Please Enter SwanID: ")
        if(swanID != ""):
            if(not df[df["SWANID"] == int(swanID)].empty):
                row = df[df["SWANID"] == int(swanID)]
                st.write(row)
                with st.container(border=True):
                    st.image("OneDrive/Documents/Work/CMU_grad/spring2024/healthIS/his_finalproject_variables_page-0001.jpg")
                    st.image("OneDrive/Documents/Work/CMU_grad/spring2024/healthIS/his_finalproject_variables_page-0002.jpg")
                    st.image("OneDrive/Documents/Work/CMU_grad/spring2024/healthIS/his_finalproject_variables_page-0003.jpg")
                    st.image("OneDrive/Documents/Work/CMU_grad/spring2024/healthIS/his_finalproject_variables_page-0004.jpg")
            else:
                st.write("Not a valid SwanID. Please enter again.")

    response = False
    if choice == "Treatment Options":
        if response == False:
            st.write("Please enter information about the folling questions regarding menopausal symptoms.")
            
            # formulate questions
            question1 = st.radio('Please indicate your race',
                    ['American Indian or Alaska Native',
                    'Asian',
                    'Black or African American',
                    'Native Hawaiian or Other Pacific Islander',
                    'White'])
            question2 = st.radio('Have you been using estrogen/progestin pills prior?',
                    ['Yes',
                    'No']) # 1 no
            question3 = st.radio('Have you been experiencing hot flashes in the past two weeks?',
                    ['Every day',
                    '9-13 days',
                    '6-8 days',
                    '1-5 days',
                    'Not at all']) #1 not at all
            question4 = st.radio('Have you been experiencing vaginal dryness in the past two weeks?',
                    ['Every day',
                    '9-13 days',
                    '6-8 days',
                    '1-5 days',
                    'Not at all']) # 1 not at all
            question5 = st.radio('Have you been experiencing vaginal and pelvic pain in intercourse?',
                    ['Always',
                    'Almost Always',
                    'Sometimes',
                    'Almost Never',
                    'Never',
                    'No intercourse in the past 6 months']) # 1 always
            question6 = st.radio('Have you been experiencing frequent mood changes in the past two weeks?',
                    ['Every day',
                    '9-13 days',
                    '6-8 days',
                    '1-5 days',
                    'Not at all']) # 1 not at all
            question7 = st.radio('What was the outcome of your first pregnancy?',
                    ['Live Birth',
                    'Still Birth',
                    'Miscarriage',
                    'Abortion',
                    'Tubal(Ectopic) Pregnancy',
                    'Still Pregnant',
                    'N/A']) #1-6, -1 N/A
            question8 = st.radio('What was the outcome of your second pregnancy?',
                    ['Live Birth',
                    'Still Birth',
                    'Miscarriage',
                    'Abortion',
                    'Tubal(Ectopic) Pregnancy',
                    'Still Pregnant',
                    'N/A']) #1-6, -1 N/A
            question9 = st.radio('What was the outcome of your third pregnancy?',
                    ['Live Birth',
                    'Still Birth',
                    'Miscarriage',
                    'Abortion',
                    'Tubal(Ectopic) Pregnancy',
                    'Still Pregnant',
                    'N/A']) #1-6, -1 N/A
            question10 = st.radio('Are you currently taking progestin pill #1 (ex. Provera)?',
                    ['Yes',
                    'No']) # 1 no
            question11 = st.radio('What is your current menopausal status?',
                    ['Hysterectomy/both ovaries removed',
                    'Post-menopausal',
                    'Late perimenopause',
                    'Early perimenopause',
                    'Pre-menopausal',
                    'Pregnant/breastfeeding',
                    'Unknown due to hormones (HT) use',
                    'Unknown due to hysterectomy']) # 1-8

            # compute completed
            if st.button("Submit Responses"):
                response = True
        
        if response == True:
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = getInputs(question1, question2, question3, question4,
                                                            question5, question6, question7, question8,
                                                            question9, question10, question11)
            
            # manually create data
            temp = {'RACE': [q1],
                    'PELVIC0': [q5],
                    'MOODCHG0': [q6],
                    'OUTCM20': [q9],
                    'OUTCM30': [q8],
                    'OUTCM10': [q7],
                    'PROGES15': [q10],
                    'ESTLSTV5': [q2],
                    'STATUS10': [q11],
                    'HOTFLAS10': [q3],
                    'VAGINDR10': [q4]}
            data = pd.DataFrame(temp)
            y_pred = model.predict(data)
            print(data)
            print(y_pred)
            
            if(y_pred[0] == 0):
                y_pred = 1
            else:
                y_pred = 0

            with st.container(border=True):
                if(y_pred == 1):
                    st.image('OneDrive/Documents/Work/CMU_grad/spring2024/healthIS/hrt.jpg')
                else:
                    st.write("Suggestions: " + str(y_pred))
                if st.button("Back to responses"):
                    response = False

if __name__ == "__main__":
    df = getdata()
    updated_df = createData(df)
    model = getModel(updated_df)
    main(df, model)
