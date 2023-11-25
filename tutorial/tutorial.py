
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels import api as sm


def create_model(csv_file):
    df = pd.read_csv(csv_file)
    Y = df['Relative Price']
    X = df.drop(['Relative Price', 'Pizza Name', 'Topping 3_Meat', 'Topping 3_None', 'Topping 4_Fish', 'Topping 4_None',
                 'Overall Weight'], axis=1)
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
    model = sm.OLS(y_train, X_train).fit()
    print("model created")
    return model


def predict_price(pizza_data_record, model):
    df = pd.DataFrame([pizza_data_record])
    print(df)
    print(df.transpose())
    predicted_price = model.predict(df)
    print(f"Predicted Price for the User's Pizza: {predicted_price.values[0]}")
    return predicted_price.values[0]


pizza_data_record = {
    'Intercept': 2,                     #
    'Topping 1': 0,                     # 0: no, 1: yes
    'Topping 2': 1,                     # 0: no, 1: yes
    'Topping 3': 0,                     # 0: no, 1: yes
    'Size': 1,                          # 0: Small, 1: Large, 2: Big
    'Extras Sauce': 1,                  # 0: no, 1: yes
    'Extra Cheese': 0,                  # 0: no, 1: yes
    'Distance to City Center (km)': 3,  # 1,3,5,10 km
    'Restaurant': 0,                    # 0: Take-Away, 1: Dine-In
    'Rating': 4                         # 1,2,3,4,5,6 Stars
}

# create model
model = create_model(csv_file="data/pizza_dataset_relative_price.csv")
print(model.summary())

# predict price
price = predict_price(pizza_data_record, model)
print(f"price: {price} €")


