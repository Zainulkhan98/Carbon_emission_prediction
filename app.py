
import numpy as np
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import seaborn as sns




all_columns = [['Entity', 'Year', 'Access to electricity (% of population)',
       'Access to clean fuels for cooking',
       'Renewable-electricity-generating-capacity-per-capita',
       'Financial flows to developing countries (US $)',
       'Renewable energy share in the total final energy consumption (%)',
       'Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)',
       'Electricity from renewables (TWh)',
       'Low-carbon electricity (% electricity)',
       'Primary energy consumption per capita (kWh/person)',
       'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
       'Renewables (% equivalent primary energy)', 'gdp_growth',
       'gdp_per_capita', 'Density(P/Km2)', 'Land Area(Km2)', 'Latitude',
       'Longitude']]


# Inital Evalution Metrics   -- droped to drop col
initial_metrics = {
    'R^2 Score': 0.9935696562978612,
    'Mean Squared Error': 2183376790.724549,
    'Root Mean Squared Error': 46726.61758274987,
    'Mean Absolute Error': 8956.88632302643,
    'Mean Absolute Percentage Error': 2.4811190642536243
}

# Utilized the drop_col by imputing null values by impute(strategy=mean)
mean_metrics = {
    'R^2 Score': 0.9931879063232545,
    'Mean Squared Error': 2312997239.183441,
    'Root Mean Squared Error': 48093.62992313474,
    'Mean Absolute Error': 9294.261663486544,
    'Mean Absolute Percentage Error': 3.5859127270207596
}

# Utilized the drop_col by imputing null values by impute(strategy=median)
median_metrics = {
    'R^2 Score': 0.995072983807074,
    'Mean Squared Error': 1672932785.7826223,
    'Root Mean Squared Error': 40901.50102114374,
    'Mean Absolute Error': 8899.436247283427,
    'Mean Absolute Percentage Error': 3.2392032115701763
}

all_median_coll = {'R^2 Score': 0.9980926709614629,
                   'Mean Squared Error': 649531208.6467444,
                   'Root Mean Squared Error': 25485.90215485307,
                   'Mean Absolute Error': 6279.11960222393,
                   'Mean Absolute Percentage Error': 0.9899216505982827}

















Data = pd.read_csv('global-data-on-sustainable-energy.csv')
county = Data['Entity'].unique().tolist()

def main():
    st.title("CO2 Emissions Prediction")  # Corrected title for clarity
    st.write("Predict the CO2 emissions of a country based on past data.")

    page = st.sidebar.selectbox("Choose a page", ["Home", "Prediction",'Visualization'])
    if page == "Home":

        st.markdown("""
            # Predicting a Sustainable Future: Energy Consumption and CO2 Emissions
            Welcome! This project tackles a critical challenge facing our world today: ensuring access to clean and affordable energy while mitigating climate change.

            Here's how we're approaching this challenge:

            ## Data Power
            We leverage the power of machine learning to analyze the vast dataset of "Global Data on Sustainable Energy." This dataset provides detailed information on energy consumption patterns for 176 countries over two decades (2000-2020).

            ## Predictive Models
            We develop machine learning models to predict two crucial aspects:
            - **Future Energy Needs**: Our models aim to forecast the energy requirements of each country in the coming years. This helps us understand the evolving demand for energy sources.
            - **Carbon Emission Levels**: By predicting carbon emission levels, we can assess the environmental impact of energy consumption and identify areas for improvement.

            ## Benefits
            - **Sustainable Development**: By accurately predicting energy needs and carbon emissions, we can guide policymakers and stakeholders towards sustainable energy solutions. This could involve promoting renewable energy sources, improving energy efficiency, and adopting cleaner technologies.
            - **Informed Decision-Making**: The insights from our models can empower governments and organizations to make data-driven decisions that balance energy security with environmental responsibility.
            - **Global Impact**: This project's focus on 176 countries allows us to analyze energy trends on a global scale, fostering international collaboration towards a sustainable energy future.
            """)


    elif page == "Prediction":

        columns = {
            'Entity': st.selectbox('Entity', county),
            'Year': st.number_input('Year', value=2012),
            'Access to electricity (% of population)': st.number_input('Access to electricity (% of population)', value=100),
            'Access to clean fuels for cooking': st.number_input('Access to clean fuels for cooking', value=100),
            'Renewable-electricity-generating-capacity-per-capita': st.number_input('Renewable-electricity-generating-capacity-per-capita', value=100),
            'Financial flows to developing countries (US $)': st.number_input('Financial flows to developing countries (US $)', value=100),
            'Renewable energy share in the total final energy consumption (%)': st.number_input('Renewable energy share in the total final energy consumption (%)', value=100),
            'Electricity from fossil fuels (TWh)': st.number_input('Electricity from fossil fuels (TWh)', value=100),
            'Electricity from nuclear (TWh)': st.number_input('Electricity from nuclear (TWh)', value=100),
            'Electricity from renewables (TWh)': st.number_input('Electricity from renewables (TWh)', value=100),
            'Low-carbon electricity (% electricity)': st.number_input('Low-carbon electricity (% electricity)', value=100),
            'Primary energy consumption per capita (kWh/person)': st.number_input('Primary energy consumption per capita (kWh/person)', value=100),
            'Energy intensity level of primary energy (MJ/$2017 PPP GDP)': st.number_input('Energy intensity level of primary energy (MJ/$2017 PPP GDP)', value=100),
            'Renewables (% equivalent primary energy)': st.number_input('Renewables (% equivalent primary energy)', value=100),
            'gdp_growth': st.number_input('gdp_growth', value=100),
            'gdp_per_capita': st.number_input('gdp_per_capita', value=100),
            'Density(P/Km2)': st.number_input('Density(P/Km2)', value=100),
            'Land Area(Km2)': st.number_input('Land Area(Km2)', value=100),
            'Latitude': st.number_input('Latitude', value=100),
            'Longitude': st.number_input('Longitude', value=100)


        }

        if st.button('Predict'):
            st.write('Model is Predicting...')

            # Create a DataFrame from user input
            data = pd.DataFrame(columns, index=[0])


            # Load the fitted transformers
            impute = joblib.load('imputer.pkl')
            scaler = joblib.load('scaler.pkl')
            ohe = joblib.load('encoder.pkl')

            scal_req_col = ['Land Area(Km2)']
            cat_col = ['Entity']

            # Transform the data
            data[scal_req_col] = scaler.fit_transform(data[scal_req_col])

            # Transform categorical columns
            # Before the main function
            ohe.fit(Data[['Entity']])  # Fit the encoder on the training data

            # Inside the main function
            data_to_encode = data[['Entity']]
            encoded_data = ohe.transform(data_to_encode)
            encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(data_to_encode.columns))
            data = pd.concat([data, encoded_df], axis=1)
            data_new = data.drop(cat_col, axis=1)

            data.to_csv('final_predict.csv', index=False)

            # Load the model
            try:
                model = joblib.load('model.pkl')
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return  # Exit function gracefully

            # Predict using the model
            st.subheader('Predicting...')
            prediction = model.predict(data_new)
            try:
                prediction = int(prediction)
                st.write(f"Predicted CO2 Emissions for {columns['Entity']} : {prediction} KT")
            except Exception as e:
                st.error(f"Error predicting: {e}")
                return
            pass

    elif page == "Visualization":



        numeric_data = Data.select_dtypes(include=['number'])

        # Display the correlation matrix
        st.subheader('Correlation Matrix')
        corr_matrix = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        # Customize the color palette
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.write("Positive Correlation: \n"
                 
                 "- There is a positive correlation between access to electricity and GDP per capita. This suggests that countries with more electrification tend to have stronger economies.\n"
                 "- A positive correlation exists between renewable energy share and access to clean fuels for cooking. This could indicate that countries investing in renewable energy are also prioritizing clean cooking solutions.\n"
                 "- Electricity consumption from renewables and fossil fuels are positively correlated. This may not be surprising as some countries might use a combination of both sources to meet their energy demands.\n"
                 
                 
                 "\nNegative Correlation:\n"
                 
                 "- Renewable energy share has a negative correlation with electricity consumption from fossil fuels. This is intuitive as countries relying more on renewables would tend to use less fossil fuel-based electricity.\n"
                 "- There's a negative correlation between the energy intensity level (MJ/$2017 PPP GDP) and GDP per capita. This implies that countries with higher economic output tend to be more energy efficient.\n")


        st.subheader("Model Accuracy Improvement")
        # Your visualization code here
        metrics = list(initial_metrics.keys())
        mean_changes = [((mean_metrics[metric] - initial_metrics[metric]) / initial_metrics[metric]) * 100 for metric in
                        metrics]
        median_changes = [((median_metrics[metric] - initial_metrics[metric]) / initial_metrics[metric]) * 100 for
                          metric in metrics]
        all_median_changes = [((all_median_coll[metric] - initial_metrics[metric]) / initial_metrics[metric]) * 100 for
                              metric in metrics]

        bar_width = 0.35
        r1 = np.arange(len(metrics))
        r2 = [x + bar_width for x in r1]
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.bar(r1, mean_changes, width=bar_width, color='b', edgecolor='grey', label='Mean Model')
        plt.bar(r2, median_changes, width=bar_width, color='r', edgecolor='grey', label='Median Model')
        plt.bar(r2, all_median_changes, width=bar_width, color='g', edgecolor='grey', label='All Median Model')

        plt.xlabel('Metrics', fontweight='bold')
        plt.xticks([r + bar_width / 2 for r in range(len(mean_changes))], metrics, rotation=90)



        plt.legend()
        plt.title('Percentage Change in Metrics from Initial Model')
        st.pyplot(fig)



        st.subheader("CO2 Emissions Yearly Trend")

        Data['Year'] = pd.to_datetime(Data['Year'], format='%Y')
        # Select the columns you want to plot
        emissions = Data['Value_co2_emissions_kt_by_country']
        year = Data["Year"]

        # Create the line plot
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        plt.plot(year, emissions, marker='o', linestyle='-')  # Customize markers and linestyle

        # Add labels and title
        plt.xlabel("Year")
        plt.ylabel("CO2 Emissions(KT)")
        plt.title("CO2 Emissions Yearly Trend By Country")

        # Rotate x-axis labels for readability if needed
        plt.xticks(rotation=45)  # Uncomment if x-axis labels are overlapping

        # Show the plot
        plt.grid(True)
        plt.tight_layout()

        # Display the plot with Streamlit
        st.pyplot(plt)

        st.subheader("Renewable Energy Share in Total Energy Consumption over the Years")

        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
        sns.lineplot(data=Data, x='Year', y='Renewable energy share in the total final energy consumption (%)', ax=ax)
        st.pyplot(fig)

        st.markdown("""
                    ## Increase in Renewable Energy Consumption
                    The most prominent insight is the steady rise in renewable energy consumption in the United States over the two decades (2000-2020). This indicates a growing shift towards cleaner energy sources like solar, wind, and hydropower.

                    ## Possible Reasons
                    While the graph doesn't directly explain the reasons behind this rise, here are some potential contributing factors:

                    - **Growing Environmental Concerns**: Increased public awareness about climate change and its impact might be driving a shift towards renewable energy sources.
                    - **Government Policies**: Supportive government policies, such as subsidies or tax breaks for renewable energy, could incentivize its adoption.
                    - **Technological Advancements**: The decreasing costs and increasing efficiency of renewable energy technologies might make them more attractive options.
                    """)

    # elif page == 'EDA Report':
    #     from streamlit.components.v1 import html
    #
    #     with open('EDA_Report_foods.html', 'r') as html_file:
    #         html_content = html_file.read()
    #         st.markdown(html_content)




if __name__ == '__main__':

    main()




















