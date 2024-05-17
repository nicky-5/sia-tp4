import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from itertools import cycle

plt.rcParams['figure.dpi'] = 200
plot_country_names = True

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('./data/europe.csv')

    # Assuming 'Country' is the column to exclude
    df_numeric = df.loc[:, df.columns != 'Country']

    scaler = StandardScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # Step 2: Transform the data to long format
    df_long = df_normalized.melt(var_name='Category', value_name='Value')

    # Step 3: Create a boxplot
    ax = sns.boxplot(data=df_long, x='Category', y='Value')

    if (plot_country_names):
        # Plot dots for each country
        for index, row in df_normalized.iterrows():
            for i, category in enumerate(df_normalized.columns):
                category_index = list(df_normalized.columns).index(category)
                ax.plot(category_index, row[category], marker='o', color='red')
                ax.annotate(df.iloc[index]['Country'], (category_index, row[category]),
                            xytext=(5, 0), textcoords='offset points', fontsize=6, ha='left')

    # Show the plot

    plt.xticks(range(len(df_numeric.columns)), df_numeric.columns,
               rotation=45)  # Rotate x-axis labels

    plt.show()

    # Separate features (excluding 'Country')
    X = df.drop(columns=['Country'])

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create DataFrame for biplot
    biplot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    biplot_df['Country'] = df['Country']

    # Get unique country names
    countries = biplot_df['Country'].unique()

    # Define a color cycle for the countries

    # Plot biplot
    plt.figure(figsize=(10, 8))

    # Plot each country with a different color
    for country in countries:
        country_data = biplot_df[biplot_df['Country'] == country]
        plt.scatter(country_data['PC1'],
                    country_data['PC2'], label=country, alpha=0.5)

    # Annotate each point with country name
    if (plot_country_names):
        for i, txt in enumerate(biplot_df['Country']):
            plt.annotate(
                txt, (biplot_df['PC1'][i], biplot_df['PC2'][i]), fontsize=8)

    for i in range(len(X.columns)):
        plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
                  color='r', alpha=0.5, head_width=0)
        plt.text(pca.components_[0, i] * 1.3, pca.components_[1, i] * 1.3,
                 X.columns[i], color='g', ha='center', va='center')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Biplot using PCA')
    plt.grid()

    # Plot legend outside the graph
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Aggregate PC1 values for each country
    pc1_by_country = biplot_df.groupby('Country')['PC1'].mean().reset_index()

    # Plot bar graph for PC1 per country
    plt.figure(figsize=(12, 6))
    plt.bar(pc1_by_country['Country'], pc1_by_country['PC1'], color='skyblue')
    plt.xlabel('Country')
    plt.ylabel('Average PC1')
    plt.title('Average PC1 per Country')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
