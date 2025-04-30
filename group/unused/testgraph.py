# testchart.py

from flask import Flask, render_template_string
import pandas as pd
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

def plot_to_base64(df, ticker):
    """
    Plots the full Adjusted Close series exactly as before,
    then returns the PNG as a base64 data URI.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df['Adj Close'], marker='o', linestyle='-')
    ax.set_title(f'{ticker} Adjusted Close — Full History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_data

@app.route('/')
def index():
    # Load & sort your data exactly as before
    amzn = pd.read_csv('AMZN.csv', parse_dates=['Date']).sort_values('Date')
    msft = pd.read_csv('MSFT.csv', parse_dates=['Date']).sort_values('Date')

    # Generate the two plots as base64-encoded PNGs
    amzn_img = plot_to_base64(amzn, 'AMZN')
    msft_img = plot_to_base64(msft, 'MSFT')

    # Simple HTML embedding the images
    html = """
    <!DOCTYPE html>
    <html>
      <head><title>Stock Charts</title></head>
      <body>
        <h2>AMZN Adjusted Close</h2>
        <img src="data:image/png;base64,{{ amzn_img }}" alt="AMZN chart"/>
        <hr/>
        <h2>MSFT Adjusted Close</h2>
        <img src="data:image/png;base64,{{ msft_img }}" alt="MSFT chart"/>
      </body>
    </html>
    """
    return render_template_string(html, amzn_img=amzn_img, msft_img=msft_img)

if __name__ == '__main__':
    app.run(debug=True)
