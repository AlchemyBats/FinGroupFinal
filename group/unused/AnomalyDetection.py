import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
# Placeholder for autoencoder implementation
# from tensorflow.keras import Model, layers

# Placeholder for Ollama anomaly detector client
# import ollama


def load_data(path: str, date_col: str = 'Date') -> pd.DataFrame:
    """
    Load CSV, parse dates, and sort.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df.sort_values(date_col, inplace=True)
    
    # Map 'price' to 'Close' if needed
    if 'price' in df.columns and 'Close' not in df.columns:
        df['Close'] = df['price']
        print("Mapped 'price' column to 'Close'")
    
    # Create synthetic Volume if missing
    if 'Volume' not in df.columns:
        # Use shares_outstanding * price if available
        if 'shares_outstanding' in df.columns and 'price' in df.columns:
            df['Volume'] = df['shares_outstanding'] * df['price'] / 100  # Scale down for reasonable values
            print("Created synthetic 'Volume' from shares_outstanding and price")
        else:
            # Create synthetic Volume based on price
            df['Volume'] = df['Close'] * np.random.normal(1000000, 200000, size=len(df))
            df['Volume'] = df['Volume'].abs().astype(int)
            print("Created random synthetic 'Volume' based on price")
    
    # Validate required columns now exist
    required_cols = ['Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Could not create required columns: {missing_cols}")
        
    return df


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Compute technical features: returns, volatility, RSI, MACD, OBV, etc.
    Also adds financial ratio features if available in the data.
    """
    def __init__(self, window_short=12, window_long=26, window_signal=9):
        self.window_short = window_short
        self.window_long = window_long
        self.window_signal = window_signal

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Returns
        df['Return'] = df['Close'].pct_change()
        # Volatility (rolling std)
        df['Volatility_5d'] = df['Return'].rolling(5).std()
        # Volume z-score
        df['Vol_Z'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
        # Momentum
        df['Momentum'] = df['Close'].diff(self.window_short)
        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.window_short).mean()
        avg_loss = loss.rolling(self.window_short).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        # MACD
        ema_short = df['Close'].ewm(span=self.window_short, adjust=False).mean()
        ema_long = df['Close'].ewm(span=self.window_long, adjust=False).mean()
        df['MACD'] = ema_short - ema_long
        df['MACD_Signal'] = df['MACD'].ewm(span=self.window_signal, adjust=False).mean()
        # OBV
        df['OBV'] = (np.sign(df['Return']) * df['Volume']).cumsum()
        
        # Add financial ratio features if available
        financial_features = [
            'pe_ratio', 'current_ratio', 'quick_ratio', 'debt_to_equity_ratio',
            'return_on_equity', 'return_on_assets', 'return_on_investment',
            'ps_ratio', 'pb_ratio', 'pfcf_ratio'
        ]
        
        for feature in financial_features:
            if feature in df.columns:
                # Convert to numeric, coerce errors to NaN
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                # Calculate z-score for the feature
                if df[feature].notna().sum() > 0:  # Only if we have non-NaN values
                    feature_mean = df[feature].mean()
                    feature_std = df[feature].std()
                    if feature_std > 0:  # Avoid division by zero
                        df[f'{feature}_Z'] = (df[feature] - feature_mean) / feature_std
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        return df


def inject_synthetic_anomalies(X: pd.DataFrame, contamination: float) -> pd.DataFrame:
    """
    Randomly mark a fraction of rows as anomalies for benchmarking.
    Adds a 'TrueLabel' column: 1=anomaly, 0=normal.
    """
    df = X.copy()
    n = len(df)
    n_anom = int(contamination * n)
    labels = np.zeros(n, dtype=int)
    labels[np.random.choice(n, n_anom, replace=False)] = 1
    df['TrueLabel'] = labels
    return df


def build_pipeline(contamination: float=0.01, k_features: int=10) -> Pipeline:
    """
    Builds a sklearn Pipeline with feature engineering, scaling, and PCA.
    """
    # Identify columns to drop - date column and any non-numeric columns
    cols_to_drop = ['Date', 'ticker']
    
    feature_union = Pipeline([
        ('feat_eng', FeatureEngineering()),
        ('drop_cols', DropColumnsTransformer(cols_to_drop)),
        ('scaler', StandardScaler())
        # Removed SelectKBest since we don't have labeled data for f_classif
    ])

    pipeline = Pipeline([
        ('features', feature_union),
        ('pca', PCA(n_components=min(k_features, 5)))
    ])

    return pipeline


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to drop non-feature columns (like 'Date').
    """
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols)


def fit_models(X: np.ndarray, contamination: float) -> dict:
    """
    Fit multiple anomaly detectors.
    """
    models = {
        'IsolationForest': IsolationForest(contamination=contamination, random_state=42).fit(X),
        'LOF': LocalOutlierFactor(contamination=contamination, novelty=True).fit(X),
        'OneClassSVM': OneClassSVM(nu=contamination, kernel='rbf').fit(X),
        'EllipticEnvelope': EllipticEnvelope(contamination=contamination).fit(X),
        # 'Autoencoder': ...  # implement autoencoder training here
        # 'Ollama': ...       # call ollama.anomaly_detector here
    }
    return models


def score_models(models: dict, X: np.ndarray) -> pd.DataFrame:
    """
    Score each model and flag anomalies.
    Returns DataFrame with score and flag columns per model.
    """
    results = {}
    for name, m in models.items():
        # Different models have different scoring methods
        if name == 'LOF':
            # For LOF with novelty=True, use negative of score_samples
            raw_scores = -m.score_samples(X)
        elif hasattr(m, 'decision_function'):
            # For IsolationForest, OneClassSVM, EllipticEnvelope
            raw_scores = -m.decision_function(X)
        elif hasattr(m, '_decision_function'):
            # Fallback for some models
            raw_scores = -m._decision_function(X)
        else:
            raise ValueError(f"Model {name} has no supported scoring method")
            
        # Get contamination parameter (nu for OneClassSVM)
        if name == 'OneClassSVM':
            contamination = m.nu
        else:
            contamination = m.contamination
            
        # Flag anomalies based on percentile threshold
        flags = (raw_scores > np.percentile(raw_scores, 100 * (1 - contamination)))
        results[f'{name}_score'] = raw_scores
        results[f'{name}_flag'] = flags.astype(int)
    return pd.DataFrame(results)


def evaluate_performance(df_scores: pd.DataFrame, df_true: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy and false positive rates per model.
    Assumes df_true['TrueLabel'] contains 0/1.
    """
    metrics = {}
    for name in [c.split('_')[0] for c in df_scores.columns if c.endswith('_flag')]:
        flag_col = f'{name}_flag'
        tp = ((df_scores[flag_col] == 1) & (df_true['TrueLabel'] == 1)).sum()
        fp = ((df_scores[flag_col] == 1) & (df_true['TrueLabel'] == 0)).sum()
        tn = ((df_scores[flag_col] == 0) & (df_true['TrueLabel'] == 0)).sum()
        fn = ((df_scores[flag_col] == 0) & (df_true['TrueLabel'] == 1)).sum()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        metrics[name] = {'accuracy': accuracy, 'false_positive_rate': fpr}
    return pd.DataFrame(metrics).T


def visualize_results(df_scores: pd.DataFrame, df_orig: pd.DataFrame = None):
    """
    Generate common anomaly-detection plots:
      - Histograms of scores
      - Time-series overlays of flags (requires 'Date')
      - Bar chart of flag rates
      - Violin plots of score distributions
    """
    # Histogram per model
    for score_col in [c for c in df_scores if c.endswith('_score')]:
        plt.figure(figsize=(10, 6))
        plt.hist(df_scores[score_col], bins=50)
        plt.title(f"Score Distribution: {score_col}")
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    # Bar chart of flag rates
    flag_rates = df_scores.filter(like='_flag').mean()
    plt.figure(figsize=(10, 6))
    flag_rates.plot(kind='bar')
    plt.title('Flag Rates by Model')
    plt.ylabel('Proportion Flagged')
    plt.tight_layout()
    plt.show()

    # Violin plots
    plt.figure(figsize=(12, 6))
    data = [df_scores[c] for c in df_scores if c.endswith('_score')]
    labels = [c.replace('_score', '') for c in df_scores.columns if c.endswith('_score')]
    plt.violinplot(data)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.title('Score Distribution Violin Plot')
    plt.tight_layout()
    plt.show()

     # Time-series overlay if Date supplied
    if df_orig is not None and 'Date' in df_orig.columns:
        # Align dates from original data with the scores dataframe index
        try:
            # Ensure df_orig has 'Date' column and index aligns with df_scores
            if not df_scores.index.isin(df_orig.index).all():
                print("Warning: Index mismatch between scores and original data. Cannot align dates accurately for time series plot.")
                dates = None # Indicate dates couldn't be aligned
            else:
                # Ensure the index exists in df_orig before attempting .loc
                aligned_index = df_scores.index[df_scores.index.isin(df_orig.index)]
                if len(aligned_index) < len(df_scores.index):
                     print(f"Warning: Some score indices ({len(df_scores.index) - len(aligned_index)}) not found in original data index. Plotting partial data.")
                if len(aligned_index) == 0:
                     print("Warning: No matching indices found between scores and original data. Skipping time series plot.")
                     dates = None
                else:
                    dates = df_orig.loc[aligned_index, 'Date']
                    # Crucially, filter df_scores to only include rows with aligned dates
                    df_scores_aligned = df_scores.loc[aligned_index] 


        except KeyError:
            print("Warning: Could not align dates based on index (KeyError). Time series plots might be inaccurate.")
            dates = None # Indicate dates couldn't be aligned
            df_scores_aligned = None
        except Exception as e:
            print(f"Warning: An unexpected error occurred during date alignment: {e}. Skipping time series plot.")
            dates = None # Indicate dates couldn't be aligned
            df_scores_aligned = None

        # Proceed only if dates were successfully aligned
        if dates is not None and df_scores_aligned is not None:
            # Ensure lengths match after alignment (final sanity check)
            if len(dates) != len(df_scores_aligned):
                 print(f"Warning: Post-alignment length mismatch ({len(dates)} dates vs {len(df_scores_aligned)} scores). Skipping time series plot.")
            else:
                for flag_col in df_scores_aligned.filter(like='_flag'):
                    plt.figure(figsize=(12, 6))
                    # Ensure flag_col data is numpy array for plotting compatibility
                    flags = df_scores_aligned[flag_col].values
                    plt.plot(dates, flags, marker='.', linestyle='none')
                    plt.title(f"Anomaly Flags Over Time: {flag_col}")
                    plt.xlabel('Date')
                    plt.ylabel('Flag (0/1)')
                    plt.tight_layout()
                    plt.show()
        else:
             print("Skipping time series plot due to date alignment issues or lack of matching data.")

def run_pipeline(
    path: str,
    contamination_levels: list = [0.005, 0.01, 0.02],
    k_features: int = 10,
    filter_ticker: str = None
):
    """
    Full end-to-end: load, feature-engineer, inject anomalies,
    build pipeline, fit models, score, evaluate, visualize.
    Returns dict of results per contamination level.
    
    Parameters:
    -----------
    path : str
        Path to the CSV file
    contamination_levels : list
        List of contamination levels to test
    k_features : int
        Number of features to select
    filter_ticker : str, optional
        If provided, only analyze data for this ticker symbol
    """
    # Load data
    df = load_data(path)
    
    # Filter by ticker if specified
    if filter_ticker and 'ticker' in df.columns:
        print(f"Filtering data for ticker: {filter_ticker}")
        df = df[df['ticker'] == filter_ticker]
        if len(df) == 0:
            raise ValueError(f"No data found for ticker {filter_ticker}")
    
    # Apply feature engineering
    fe = FeatureEngineering()
    df_feat = fe.transform(df)
    print(f"Engineered features. Data shape: {df_feat.shape}")
    
    # Print available features
    print(f"Available features: {df_feat.columns.tolist()}")

    results = {}
    for cont in contamination_levels:
        print(f"\nProcessing contamination level: {cont}")
        df_injected = inject_synthetic_anomalies(df_feat, cont)
        pipeline = build_pipeline(contamination=cont, k_features=min(k_features, len(df_feat.columns)-1))
        
        try:
            X = pipeline.fit_transform(df_injected)
            print(f"Pipeline transformed data to shape: {X.shape}")
            
            models = fit_models(X, contamination=cont)
            df_scores = score_models(models, X)
            df_metrics = evaluate_performance(df_scores, df_injected)

            print("Model performance metrics:")
            print(df_metrics)
            
            visualize_results(df_scores, df)
            results[cont] = {
                'scores': df_scores,
                'metrics': df_metrics
            }
        except Exception as e:
            print(f"Error at contamination {cont}: {e}")
            continue
            
    return results


if __name__ == '__main__':
    try:
        # Example usage:
        path = 'corporate_market_data.csv'
        
        # Run for a specific ticker (e.g., AAPL) to avoid mixing different companies
        # which could create false anomalies
        ticker = 'AAPL'
        
        # Use fewer features since we have limited data points per ticker
        k_features = 5
        
        # Use higher contamination to find more anomalies in small datasets
        contamination_levels = [0.05, 0.1]
        
        results = run_pipeline(
            path=path, 
            contamination_levels=contamination_levels,
            k_features=k_features,
            filter_ticker=ticker
        )
        
        print("\nAnomaly detection pipeline completed successfully.")
        print(f"Results for {ticker} with {k_features} features at contamination levels {contamination_levels}")
    except Exception as e:
        print(f"Error running anomaly detection pipeline: {e}")
