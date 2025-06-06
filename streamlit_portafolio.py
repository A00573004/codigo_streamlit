
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Análisis Financiero de Portafolios")

# Sidebar - Selección de parámetros
st.sidebar.header("Configuración del portafolio")

tickers = st.sidebar.text_input("Activos (separados por coma)", "AAPL, MSFT, GOOG").split(",")
tickers = [t.strip().upper() for t in tickers]
start_date = st.sidebar.date_input("Fecha de inicio", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", pd.to_datetime("2024-12-31"))
risk_free_rate = st.sidebar.number_input("Tasa libre de riesgo (%)", value=4.0) / 100
benchmark = st.sidebar.text_input("Benchmark (ej. ^GSPC)", "^GSPC")

n_portfolios = st.sidebar.slider("Número de simulaciones", 1000, 10000, 3000)

# Descarga de datos
@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data.dropna()

data = load_data(tickers, start_date, end_date)
daily_returns = data.pct_change().dropna()
mean_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()
n_assets = len(tickers)

# Simulación de portafolios
results = np.zeros((4, n_portfolios))
weights_record = []

for i in range(n_portfolios):
    weights = np.random.random(n_assets)
    weights /= np.sum(weights)
    weights_record.append(weights)

    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev

    results[0,i] = portfolio_return
    results[1,i] = portfolio_stddev
    results[2,i] = sharpe_ratio
    results[3,i] = i

# Portafolio óptimo
max_sharpe_idx = np.argmax(results[2])
opt_weights = weights_record[int(results[3, max_sharpe_idx])]
opt_return = results[0, max_sharpe_idx]
opt_std = results[1, max_sharpe_idx]

# Value at Risk (VaR)
confidence_level = 0.95
VaR = -np.percentile(np.dot(daily_returns, opt_weights), 100 * (1 - confidence_level))

# Cálculo de CAL
CAL_x = np.linspace(0, 0.4, 100)
CAL_y = risk_free_rate + ((opt_return - risk_free_rate) / opt_std) * CAL_x

# Gráficos
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(results[1,:], results[0,:], c=results[2,:], cmap="viridis", alpha=0.5)
ax.scatter(opt_std, opt_return, marker="*", color="r", s=300, label="Máx. Sharpe")
ax.plot(CAL_x, CAL_y, linestyle='--', color='gray', label="Capital Allocation Line")
ax.set_xlabel("Riesgo (Desviación Estándar Anualizada)")
ax.set_ylabel("Retorno Esperado Anualizado")
ax.set_title("Frontera eficiente y CAL")
ax.legend()
fig.colorbar(scatter, label="Sharpe Ratio")

st.pyplot(fig)

# Resultados
st.subheader("Resultados del Portafolio Óptimo")
st.write(f"**Sharpe Ratio:** {results[2, max_sharpe_idx]:.2f}")
st.write(f"**Retorno esperado:** {opt_return:.2%}")
st.write(f"**Riesgo (volatilidad):** {opt_std:.2%}")
st.write(f"**Value at Risk (95% diario):** {VaR:.2%}")
st.write("**Pesos del portafolio óptimo:**")
st.dataframe(pd.DataFrame({"Ticker": tickers, "Peso": opt_weights}).set_index("Ticker"))

# Benchmark
benchmark_data = yf.download(benchmark, start=start_date, end=end_date)["Adj Close"].pct_change().dropna()
benchmark_return = benchmark_data.mean() * 252
benchmark_std = benchmark_data.std() * np.sqrt(252)

st.subheader("Comparación con el Benchmark")
st.write(f"**Benchmark ({benchmark}) retorno anual:** {benchmark_return:.2%}")
st.write(f"**Benchmark ({benchmark}) riesgo anual:** {benchmark_std:.2%}")
