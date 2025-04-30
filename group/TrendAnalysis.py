import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
import ollama

# === Ollama anomaly detector integration ===
class OllamaAnomalyDetector(BaseEstimator):
    """
    Wraps an Ollama model’s built-in anomaly-detector
    so it can be slotted into our sklearn scoring pipeline.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ollama.Ollama()

    def fit(self, X, y=None):
        # No training needed on client side
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # Call Ollama’s anomaly detector processor
        response = self.client.call(
            model=self.model_name,
            processor="anomaly-detector",
            data=X.tolist()
        )
        # Expecting: response = {'scores': [float, float, …]}
        return np.array(response['scores'])


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
        if 'shares_outstanding' in df.columns and 'price' in df.columns:
            df['Volume'] = df['shares_outstanding'] * df['price'] / 100  
            print("Created synthetic 'Volume' from shares_outstanding and price")
        else:
            df['Volume'] = df['Close'] * np.random.normal(1000000, 200000, size=len(df))
            df['Volume'] = df['Volume'].abs().astype(int)
            print("Created random synthetic 'Volume' based on price")
    
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
        df['Return'] = df['Close'].pct_change()
        df['Volatility_5d'] = df['Return'].rolling(5).std()
        df['Vol_Z'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
        df['Momentum'] = df['Close'].diff(self.window_short)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.window_short).mean()
        avg_loss = loss.rolling(self.window_short).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        ema_short = df['Close'].ewm(span=self.window_short, adjust=False).mean()
        ema_long = df['Close'].ewm(span=self.window_long, adjust=False).mean()
        df['MACD'] = ema_short - ema_long
        df['MACD_Signal'] = df['MACD'].ewm(span=self.window_signal, adjust=False).mean()
        df['OBV'] = (np.sign(df['Return']) * df['Volume']).cumsum()
        
        financial_features = [
            'pe_ratio', 'current_ratio', 'quick_ratio', 'debt_to_equity_ratio',
            'return_on_equity', 'return_on_assets', 'return_on_investment',
            'ps_ratio', 'pb_ratio', 'pfcf_ratio'
        ]
        for feature in financial_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                if df[feature].notna().sum() > 0:
                    feature_mean = df[feature].mean()
                    feature_std = df[feature].std()
                    if feature_std > 0:
                        df[f'{feature}_Z'] = (df[feature] - feature_mean) / feature_std
        df.dropna(inplace=True)
        return df


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Transformer to drop non-feature columns (like 'Date')."""
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols)


def inject_synthetic_anomalies(X: pd.DataFrame, contamination: float) -> pd.DataFrame:
    df = X.copy()
    n = len(df)
    n_anom = int(contamination * n)
    labels = np.zeros(n, dtype=int)
    labels[np.random.choice(n, n_anom, replace=False)] = 1
    df['TrueLabel'] = labels
    return df


def build_pipeline(contamination: float=0.01, k_features: int=10) -> Pipeline:
    cols_to_drop = ['Date', 'ticker']
    feature_union = Pipeline([
        ('feat_eng', FeatureEngineering()),
        ('drop_cols', DropColumnsTransformer(cols_to_drop)),
        ('scaler', StandardScaler())
    ])
    pipeline = Pipeline([
        ('features', feature_union),
        ('pca', PCA(n_components=min(k_features, 5)))
    ])
    return pipeline


def fit_models(X: np.ndarray, contamination: float) -> dict:
    """Fit anomaly detectors: sklearn + Ollama models."""
    return {
        'OneClassSVM':      OneClassSVM(nu=contamination, kernel='rbf').fit(X),
        'EllipticEnvelope': EllipticEnvelope(contamination=contamination).fit(X),
        'Mistral':          OllamaAnomalyDetector('mistral').fit(X),
        'Deepseek-R1':      OllamaAnomalyDetector('deepseek-r1').fit(X),
        'Phi-4':            OllamaAnomalyDetector('phi-4').fit(X),
    }


def score_models(models: dict, X: np.ndarray) -> pd.DataFrame:
    results = {}
    for name, m in models.items():
        if hasattr(m, 'decision_function'):
            raw_scores = -m.decision_function(X)
        else:
            raise ValueError(f"Model {name} has no supported scoring method")
        contamination = getattr(m, 'nu', getattr(m, 'contamination', None))
        flags = (raw_scores > np.percentile(raw_scores, 100 * (1 - contamination)))
        results[f'{name}_score'] = raw_scores
        results[f'{name}_flag'] = flags.astype(int)
    return pd.DataFrame(results)


def evaluate_performance(df_scores: pd.DataFrame, df_true: pd.DataFrame) -> pd.DataFrame:
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
    for score_col in [c for c in df_scores if c.endswith('_score')]:
        plt.figure(figsize=(10, 6))
        plt.hist(df_scores[score_col], bins=50)
        plt.title(f"Score Distribution: {score_col}")
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    flag_rates = df_scores.filter(like='_flag').mean()
    plt.figure(figsize=(10, 6))
    flag_rates.plot(kind='bar')
    plt.title('Flag Rates by Model')
    plt.ylabel('Proportion Flagged')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    data = [df_scores[c] for c in df_scores if c.endswith('_score')]
    labels = [c.replace('_score', '') for c in df_scores.columns if c.endswith('_score')]
    plt.violinplot(data)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.title('Score Distribution Violin Plot')
    plt.tight_layout()
    plt.show()

    if df_orig is not None and 'Date' in df_orig.columns:
        try:
            aligned_index = df_scores.index[df_scores.index.isin(df_orig.index)]
            if aligned_index.empty:
                print("Skipping time series plot due to no matching indices.")
                return
            dates = df_orig.loc[aligned_index, 'Date']
            df_scores_aligned = df_scores.loc[aligned_index]
        except Exception:
            print("Skipping time series plot due to alignment issues.")
            return
        for flag_col in df_scores_aligned.filter(like='_flag'):
            plt.figure(figsize=(12, 6))
            plt.plot(dates, df_scores_aligned[flag_col].values, marker='.', linestyle='none')
            plt.title(f"Anomaly Flags Over Time: {flag_col}")
            plt.xlabel('Date')
            plt.ylabel('Flag (0/1)')
            plt.tight_layout()
            plt.show()


def run_pipeline(
    path: str,
    contamination_levels: list = [0.005, 0.01, 0.02],
    k_features: int = 10,
    filter_ticker: str = None
):
    df = load_data(path)
    if filter_ticker and 'ticker' in df.columns:
        df = df[df['ticker'] == filter_ticker]
        if df.empty:
            raise ValueError(f"No data found for ticker {filter_ticker}")

    fe = FeatureEngineering()
    df_feat = fe.transform(df)
    print(f"Engineered features. Data shape: {df_feat.shape}")

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
            results[cont] = {'scores': df_scores, 'metrics': df_metrics}
        except Exception as e:
            print(f"Error at contamination {cont}: {e}")
            continue
    return results


if __name__ == '__main__':
    try:
        path = 'corporate_market_data.csv'
        ticker = 'AAPL'
        k_features = 5
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
