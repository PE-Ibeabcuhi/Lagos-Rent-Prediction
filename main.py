# Importing packages
import pandas as pd
from flask import Flask, render_template, request
import pickle
import sklearn
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load the saved prediction model
model = pickle.load(open('xgb_model2.pkl','rb'))

# Load the rent data CSV file
df = pd.read_csv('Cleaned_lagos_renewed.csv')

# Home route to render the index.html file
@app.route('/')
def home():
    # Get unique locations from the CSV data
    Neighborhoods = df['Neighborhood'].unique()
    return render_template('index.html', Neighborhoods=Neighborhoods)


# Route to handle form submission and predict rent
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    toilets = int(request.form['toilets'])
    furnished = int(request.form['furnished'])
    newly_built= int(request.form['newly_built'])
    serviced = int(request.form['serviced'])
    neighborhood = request.form['Neighborhood']

    print (bedrooms,bathrooms,toilets,furnished,newly_built,serviced,neighborhood)
    input=pd.DataFrame([[serviced,newly_built,furnished,bedrooms,bathrooms,toilets,neighborhood]],columns=['Serviced','Newly Built','Furnished','Bedrooms','Bathrooms','Toilets','Neighborhood'])
    prediction= model.predict(input)[0]

    return f'₦{prediction:,.2f}'

if __name__ == '__main__':
    app.run(debug=True)

