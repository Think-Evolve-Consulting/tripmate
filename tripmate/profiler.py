import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

logger = logging.getLogger(__name__)

trip_level_features = \
    ['TripDistance', 'Quantity', 'VelocityMean', 'VelocitySD',
     'VelocityExIdleMean', 'VelocityExIdleSD', 'AccelerationMean',
     'AccelerationSD', 'DecelerationMean', 'DecelerationSD',
     'AccelerationTimePercent', 'DecelerationTimePercent', 'IdlingTimePercent',
     'CruisingTimePercent', 'FastAccelerationCount',
     'RelativePositiveAccelerationMean', 'RelativePositiveAccelerationSD',
     'VASquaredMean', 'VASquaredSD'
     ]

def get_scaled_features(
    movement_params_df, 
    features
    ):
    logger.info("Getting select set of features ...")
    features_df = movement_params_df[features].copy().reset_index(drop=True)

    logger.info(f"Selected {features_df.shape[1]} features for analysis")
    logger.info("Column names of selected features:")
    for column_name in features_df.columns.values:
        logger.info(f"{column_name}")
    
    scaler = StandardScaler().set_output(transform="pandas")
    features_scaled_df = scaler.fit_transform(features_df)
    logger.info("Scaled the features using StandardScaler")

    return features_scaled_df

def get_principal_components(
    movement_params_df, 
    features_scaled_df, 
    n_principal_components=4, 
    show_scree_plot=False, 
    random_state=42
    ):
    logger.info(f"Decomposing scaled features into {n_principal_components} principal components ...")

    # perform pca
    n_features = features_scaled_df.shape[1]
    decomposer = PCA(n_components=n_features, random_state=random_state)
    principal_components = decomposer.fit_transform(features_scaled_df)
    explained_variance = round(sum(decomposer.explained_variance_ratio_[0:n_principal_components]), 4)*100
    logger.info(f"Variance explained by {n_principal_components} principal components: {explained_variance} %")

    # display explained variance per principal component in scree plot
    if show_scree_plot is True:
        fig, ax = plt.subplots(figsize=(3, 3))
        x_data = [i+1 for i in range(len(decomposer.explained_variance_ratio_))]
        sns.scatterplot(
            x=x_data[:n_principal_components],
            y=decomposer.explained_variance_ratio_[:n_principal_components], 
            color='green'
            )
        sns.scatterplot(
            x=x_data[n_principal_components:],
            y=decomposer.explained_variance_ratio_[n_principal_components:], 
            color='blue'
            )
        ax.set_xticks(x_data[0::2])
        plt.title("Scree Plot")
        plt.xlabel("Principal Components")
        plt.ylabel("Explained Variance")
        plt.show()

    # store all principal components in a data frame
    principal_components_df = \
        pd.DataFrame(
            data=principal_components,
            columns=["PC" + str(x+1) for x in range(n_features)]
            )
        
    # add requested number of principal components alongside original data
    selected_components = ["PC" + str(x+1) for x in range(n_principal_components)]
    movement_params_df[selected_components] = principal_components_df[selected_components].values
    logger.info(f"Added {n_principal_components} principal components to movement parameters")

    # get the loadings of each feature for the requested principal components
    loadings_df = \
        pd.DataFrame(
            data=decomposer.components_[0:n_principal_components,:].T,
            columns=selected_components,
            index=features_scaled_df.columns.values
        )
    
    # get correlations between features for the requested principal components
    correlations_df = \
        pd.DataFrame(
            data=decomposer.components_[0:n_principal_components,:].T\
                /np.sqrt(decomposer.explained_variance_[0:n_principal_components]),
            columns=selected_components,
            index=features_scaled_df.columns.values
        )

    return movement_params_df, loadings_df, correlations_df

def get_tsne(
    movement_params_df, 
    features_scaled_df, 
    perplexity, 
    learning_rate=None, 
    n_dimensions_out=2, 
    initialization='random', 
    n_iterations=1000, 
    plot_embeddings=False
    ):
    logger.info("Computing low dimensional embeddings for given features using t-SNE ...")

    # intialize t-SNE with given hyper parameters 
    tsne_object = \
        TSNE(
            n_components=n_dimensions_out,
            init=initialization,
            learning_rate=(features_scaled_df.shape[0]/12) if learning_rate is None else learning_rate,
            perplexity=perplexity,
            n_iter=n_iterations
            )

    # get tsne embeddings
    tsne_embeddings = tsne_object.fit_transform(features_scaled_df)

    # report tsne attributes
    logger.debug(f"KL Divergence            : {tsne_object.kl_divergence_}")
    logger.debug(f"Effective Learning Rate  : {tsne_object.learning_rate_}")
    logger.debug(f"Number of iterations run : {tsne_object.n_iter_}")

    # add tsne embeddings to original data frame
    embeddings_titles = ["TSNE" + str(x+1) for x in range(n_dimensions_out)]
    movement_params_df[embeddings_titles] = tsne_embeddings
    logger.info(f"Added {n_dimensions_out} dimensional t-SNE embeddings to movement parameters")

    # build a data frame for tsne
    tsne_df = pd.DataFrame(data=tsne_embeddings, columns=embeddings_titles)

    # plot the embeddings
    if n_dimensions_out == 2:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
        ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
        ax.set_title(f"t-SNE for Perplexity: {perplexity}, Learning Rate: {learning_rate}")
        plt.show()

    return movement_params_df, tsne_df

def perform_kmeans_clustering(
    movement_params_df, 
    features_df, 
    data_tag, 
    n_clusters=None, 
    n_max_clusters=15, 
    random_state=42
    ):
    logger.info(f"Performing K-Means clustering based on {data_tag.lower()} data ...")

    # initialize kwargs
    kmeans_kwargs = \
        dict(
            init="random",
            n_init=10,
            max_iter=300,
            random_state=random_state
            )

    # find optimal number of clusters if n_clusters value is not specified
    if n_clusters is None:
        logger.info("Finding optimal number of clusters using knee locator and silhouette score")
        sse = []
        silhouette_coefficients = []

        # compute sum of squared error and silhouette score
        for k in range(2, n_max_clusters):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(features_df)
            sse.append(kmeans.inertia_)
            silhouette_coefficient = silhouette_score(features_df, kmeans.labels_)
            silhouette_coefficients.append(silhouette_coefficient)

        # locate the knee in elbow plot
        kneedle = \
            KneeLocator(
                x=range(2, n_max_clusters),
                y=sse,
                curve='convex',
                direction='decreasing'
            )

        logger.info(f"Optimal number of cluskers (k) as per KneeLocator: {kneedle.elbow}")
        logger.info(f"Optimal number of cluskers (k) as per Silhouette Score: {np.argmax(silhouette_coefficients)+2}")

        # plot elbow plot and silhouette score plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
        axes[0].plot(range(2, n_max_clusters), sse, marker='o', color='purple')
        axes[0].set_xticks(range(2, n_max_clusters, 2))
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("Sum of Squared Errors (SSE)")

        axes[1].plot(range(2, n_max_clusters), silhouette_coefficients, marker='o', color='green')
        axes[1].set_xticks(range(2, n_max_clusters, 2))
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Silhouette Score")

        plt.tight_layout()
        plt.show()

        logger.info(" ")
        logger.info(f"Using the maximum number of clusters among the 2 methods and proceeding further")

        n_clusters = max(kneedle.elbow, np.argmax(silhouette_coefficients) + 2)
    
    # perform k-means clustering
    logger.info(f"Running K-Means clustering for {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, **kmeans_kwargs)
    kmeans.fit(features_df)

    # report k-means clustering attributes
    kmeans_silhouette = silhouette_score(features_df, kmeans.labels_)
    logger.info(f"Silhouette Score: {round(kmeans_silhouette, 2)}")
    logger.info(f"Lowest value of Sum of Squared Errors (SSE): {round(kmeans.inertia_, 2)}")
    logger.info(f"Number of iterations required to converge: {kmeans.n_iter_}")

    # assign k-means cluster lables to movement parameters data
    movement_params_df[f"{data_tag}KMeansCluster"] = kmeans.labels_

    return movement_params_df

def perform_dbscan_clustering(
    movement_params_df, 
    features_df, 
    data_tag, 
    cluster_density=2
    ):
    logger.info(f"Performing DBSCAN clustering based on {data_tag.lower()} data ...")

    # perform DBSCAN clustering
    dbscan = DBSCAN(eps=cluster_density)
    dbscan.fit(features_df)

    # report DBSCAN attributes
    dbscan_silhouette = silhouette_score(features_df, dbscan.labels_)
    print("Silhouette Score:", round(dbscan_silhouette, 2))
    print("Number of clusters found by DBSCAN:", len(np.unique(dbscan.labels_)))

    # assign DBSCAN cluster lables to movement parameters data
    movement_params_df[f"{data_tag}DBSCANCluster"] = dbscan.labels_

    return movement_params_df