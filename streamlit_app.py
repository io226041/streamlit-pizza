import streamlit as st
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
    return model


def predict_price(pizza_data_record, model):
    df = pd.DataFrame([pizza_data_record])
    predicted_price = model.predict(df)
    return predicted_price.values[0]


def sidebar():
    st.sidebar.write("Toppings")
    topping1 = st.sidebar.selectbox(label=f"#1 - sauce", options=(["no sauce", "Tomato Sauce", "Pesto", "Alfredo"]), label_visibility="collapsed")
    topping2 = st.sidebar.selectbox(label=f"#2 - vegetable", options=(["no vegetable", "Pepperoni", "Mushrooms", "Onions", "Tomatoes"]), label_visibility="collapsed")
    topping3 = st.sidebar.selectbox(label=f"#3 - meat", options=(["no meat", "Sausage", "Bacon", "Gyros"]), label_visibility="collapsed")
    size_labels = {0: "Small", 1: "Large", 2: "Big"}
    pizza_size = st.sidebar.select_slider("Size", [0, 1, 2], 1, format_func=lambda x: size_labels[x])
    extra_sauce = st.sidebar.toggle("Extra Sauce")
    extra_cheese = st.sidebar.toggle("Extra Cheese")
    distance = st.sidebar.select_slider("Distance from City Center", [1, 3, 5, 10], 3, format_func=lambda x: f"{x}km")
    location_labels = {0: "Take-Away", 1: "Dine-In"}
    location_choice = st.sidebar.radio("Location", [0, 1], format_func=lambda x: location_labels[x])
    rating = st.sidebar.select_slider("Restaurant Raiting", [1, 2, 3, 4, 5, 6], 4, format_func=lambda x: f"{x * '★'}")

    pizza_data_record = {
        'Intercept': 2,                     #
        'Topping 1': int(not topping1.startswith("no")),  # 0: no, 1: yes
        'Topping 2': int(not topping2.startswith("no")),  # 0: no, 1: yes
        'Topping 3': int(not topping3.startswith("no")),  # 0: no, 1: yes
        'Size': pizza_size,                               # 0: Small, 1: Large, 2: Big
        'Extras Sauce': int(extra_sauce),                 # 0: no, 1: yes
        'Extra Cheese': int(extra_cheese),                # 0: no, 1: yes
        'Distance to City Center (km)': distance,         # 1,3,5,10 km
        'Restaurant': location_choice,                    # 0: Take-Away, 1: Dine-In
        'Rating': rating                                  # 1,2,3,4,5,6 Stars
    }
    return pizza_data_record


def generate_pizza_image(toppings, img_gen_model):
    if img_gen_model == 'picsum':
        seed = "-".join(map(str, toppings))
        return f"https://picsum.photos/seed/{seed}/200"
    return ""


###################
#    App Start    #
###################

st.title("Pizza Price Predictor")

# create model
model = create_model(csv_file="pizza_dataset_relative_price.csv")

# load sidebar and get input data
pizza_data_record = sidebar()

# predict price
price = predict_price(pizza_data_record, model)
st.header(f"price: {price:.2f} €")

# generate pizza image
toppings = [pizza_data_record["Topping 1"], pizza_data_record["Topping 2"], pizza_data_record["Topping 3"]]
st.image(generate_pizza_image(toppings, img_gen_model="picsum"), width=400)
