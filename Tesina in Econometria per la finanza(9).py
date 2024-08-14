# Tesina in econometria
# 9 Predire l'imprevedibile
# Predire il prezzo di una criptovaluta appare estremamente complesso a causa di variabili in gioco
# esempio, diversi studi utilizzano il prezzo del BTC come fattore guida, sfruttando le sue relazioni
# con altri mercati. Un altro fattore cruciale `e la valutazione dei futures legati alla stessa criptovaluta.
# Sfruttando strumenti statistici acquisiti durante il corso di statistica, tenta di sviluppare un modello
# di previsione per i prezzi di una criptovaluta diversa dal BTC. Utilizza un dataset in-sample
# sufficientemente ampio per sviluppare previsioni out-of-sample adeguate, fornisci stime intervallo
# appropriate e utilizza una finestra mobile di dimensioni adeguate.
# 27/03/2024
# La criptovaluta utilizzata sarà Ethereum (ETH)

# Pacchetti utilizzati
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import pmdarima as pm
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from arch import arch_model
from statsmodels.tsa.api import ARDL
from prettytable import PrettyTable 
import warnings

warnings.filterwarnings("ignore")

# Main 
def main():
    # Importo il dataset come un dict
    # Date
    time: pd.DataFrame = pd.read_excel(r"C:\Users\franc\OneDrive\Desktop\materie\econometria per la finanza\tesina\script\dati\dati.xlsx", sheet_name = 'ETH')
    names = ['BTC', 'ETH', 'ETH FUTURES', 'TETHER', 'SOLANA', 'BNB', 'XRP', 'USD COIN', 'CARDANO', 'AVALANCHE', 'DOGE', 'SP500', 'NASDQ', 'VIX', 'GOLD']
    library: pd.DataFrame = pd.read_excel(r"C:\Users\franc\OneDrive\Desktop\materie\econometria per la finanza\tesina\script\dati\dati.xlsx", sheet_name = names, index_col = 'Date')
    # Definizione delle singole serie prezzi
    dates = time['Date'].to_frame()
    BTC = library['BTC']['Ultimo prz'].to_frame()
    ETH = library['ETH']['Ultimo prz'].to_frame()
    ETH_futures = library['ETH FUTURES']['Ultimo prz'].to_frame()
    TETHER = library['TETHER']['Ultimo prz'].to_frame()
    SOLANA = library['SOLANA']['Ultimo prz'].to_frame()
    BNB = library['BNB']['Ultimo prz'].to_frame()
    XRP = library['XRP']['Ultimo prz'].to_frame()
    USD_COIN = library['USD COIN']['Ultimo prz'].to_frame()
    CARDANO = library['CARDANO']['Ultimo prz'].to_frame()
    AVALANCHE = library['AVALANCHE']['Ultimo prz'].to_frame()
    DOGE = library['DOGE']['Ultimo prz'].to_frame()
    SP500 = library['SP500']['Ultimo prz'].to_frame()
    NASDQ = library['NASDQ']['Ultimo prz'].to_frame()
    VIX = library['VIX']['Ultimo prz'].to_frame()
    GOLD = library['GOLD']['Ultimo prz'].to_frame()

    # Analisi qualitativa delle serie dei (log)prezzi
    fig, (ax1) = plt.subplots(1)
    ax1.plot(np.log(BTC), label = 'BTC', color = 'green')
    ax1.plot(np.log(ETH), label = 'ETH', color = 'purple')
    ax1.plot(np.log(SOLANA), label = 'SOLANA', color = 'orange')
    ax1.plot(np.log(BNB), label = 'BNB', color = 'cyan')
    ax1.plot(np.log(CARDANO), label = 'CARDANO', color = 'blue')
    ax = plt.gca()
    ax1.set_title('Andamento dei log prezzi delle maggiori criptovalute')
    ax1.legend(['BTC', 'ETH', 'SOLANA', 'BNB', 'CARDANO'], loc = 'lower left')
    plt.xlabel("Data")
    plt.ylabel("Log-prezzi")

    # Verifica fatti stilizzati (Distribuzione, Q-plot, ACF e pACF)
    # Istogramma per osservazione della distribuzione approssimativa
    eth_log_ret = np.log(ETH).diff().dropna()
    plt.figure()
    plt.title('Distribuzione dei log-ret ETH')
    plt.hist(eth_log_ret, bins = 50, color = 'red')
    plt.grid(linestyle ='dashed')
    plt.show
    
    # AdFuller
    adfuller_test(eth_log_ret, signif=0.05, name='ETH log returns', verbose=False)

    # ACF e pACF
    T = len(eth_log_ret)
    # Tre subplot, uno per ogni dato da rappresentare
    fig, axs = plt.subplots(1, 3, figsize = (16, 9))
    # Rendimenti logaritmici della serie dei prezzi
    axs[0].plot(eth_log_ret, marker = 'o', linestyle = '-', color = 'purple')
    axs[0].set_xlabel('Data')
    axs[0].set_ylabel('')
    axs[0].set_title(r'ETH log returns')
    axs[0].grid(True)
    # Autocorrelation function con 50 lags
    acf_eth = sm.tsa.acf(eth_log_ret, nlags = 50)[1:]
    Lag = range(1, 51)
    axs[1].stem(Lag, acf_eth, markerfmt = 'ro', linefmt = 'k-', basefmt = 'k--', label = "ACF")
    axs[1].axhline(y= 1.96 / np.sqrt(T), color = 'blue' , linestyle = '--', label = 'C.I.')
    axs[1].axhline(y= -1.96 / np.sqrt(T), color = 'blue', linestyle = '--', label = 'C.I.')
    axs[1].axis(ymin = -1, ymax = 1)
    axs[1].set_xlabel('Lag')
    axs[1].set_ylabel('')
    axs[1].set_title(r'ACF, ETH log-returns')
    axs[1].grid(True)

    # Partial autocorrelation function con 50 lags
    pacf_eth = sm.tsa.pacf(eth_log_ret, nlags= 50)[1:]
    axs[2].stem(Lag, pacf_eth, markerfmt = 'ro', linefmt = 'k-', basefmt = 'k--', label = "pACF")
    axs[2].axhline(y = 1.96 / np.sqrt(T), color = 'blue', linestyle = '--', label = 'C.I.')
    axs[2].axhline(y = -1.96 / np.sqrt(T), color = 'blue', linestyle = '--', label = 'C.I.')
    axs[2].axis(ymin = -1,ymax = 1)
    axs[2].set_xlabel('Lag')
    axs[2].set_ylabel('')
    axs[2].set_title(r'pACF, ETH log-returns')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Scelta di test set e training set
    # Iniziamo con il ridurre l'entitÃ  del campione considerato:
    # Prendiamo in considerazione il periodo che va  dal 2020 alla fine del 2023 
    inizio = '2020-01-01'
    fine = '2023-12-31'
    fin = '2023-04-01'
    training_set = t_set(eth_log_ret, inizio, fine, fin, True)
    test_set = t_set(eth_log_ret, inizio, fine, fin, False)
    
    # Regressioni lineari (anche multivariate)
    
    # Iniziamo verificando un modello semplice in cui l'unico regressore è BTC
    ETH_BTC = pd.merge(training_set, np.log(BTC).diff().dropna(), how = 'inner', left_index = True, right_index = True)
    ETH_BTC.columns = ['ETH','BTC']
    # Regressore con aggiunta di costante
    BTC_regr = ETH_BTC.loc[:, ['BTC']]
    BTC_regr.insert(0, 'constant', 1, allow_duplicates = True)
    # Stime OLS del modello univariato
    coeff: pd.DataFrame = np.linalg.inv(BTC_regr.transpose() @ BTC_regr) @ BTC_regr.transpose() @ training_set
    coeff.index = BTC_regr.columns
    
    # Generalizziamo il forecasting in rolling window delle regressioni
    # con una funzione apposita
    ETH_BTC = pd.merge(eth_log_ret, np.log(BTC).diff().dropna(), how = 'inner', left_index = True, right_index = True)
    ETH_BTC.columns = ['ETH','BTC']
    ETH_BTC = ETH_BTC[inizio:fine]
    BTC_regr = ETH_BTC.loc[:, ['BTC']]
    BTC_regr.insert(0, 'constant', 1, allow_duplicates = True)
    # Dimensione della rolling window
    window = len(t_set(ETH_BTC, inizio, fine, fin, True))
    # Rolling window forecasting
    forecast = rolling_forecast_r(2, ETH_BTC.ETH, BTC_regr, window)
    # Plot
    myFmt = mdates.DateFormatter('%m-%y')
    date_forecast = dates[len(eth_log_ret[:fin])+1:len(eth_log_ret[:fine])]
    fig, (ax1) = plt.subplots(1)
    ax1.plot(eth_log_ret[inizio:fine], label = 'ETH returns', color = 'purple', linewidth = 0.5)
    ax1.plot(date_forecast, forecast, label = 'ETH forecast', color = 'green', linewidth = 0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(myFmt)
    ax.get_xaxis().set_visible(True)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Log-ret')
    ax1.set_title('Forecast del prezzo ETH con BTC')
    
    # Forecasting accuracy
    simple_regr_rmse = rmse(eth_log_ret[date_forecast], forecast)
    
    # Contro ETH-futures
    inizio2 = '07-01-2021';
    ETH_futures = pd.merge(eth_log_ret, np.log(ETH_futures).diff().dropna(), how = 'inner', left_index = True, right_index = True)
    ETH_futures.columns = ['ETH','Futures']
    ETH_futures = ETH_futures[inizio2:fine]
    futures_regr = ETH_futures.loc[:, ['Futures']]
    futures_regr.insert(0, 'constant', 1, allow_duplicates = True)
    # Dimensione della rolling window
    window = len(t_set(ETH_futures, inizio2, fine, fin, True))
    # Rolling window forecasting
    forecast = rolling_forecast_r(2, ETH_futures.ETH, futures_regr, window)
    # Plot
    fig, (ax1) = plt.subplots(1)
    ax1.plot(eth_log_ret[inizio:fine], label = 'ETH returns', color = 'orange', linewidth = 0.5)
    ax1.plot(date_forecast[85:], forecast, label = 'ETH forecast', color = 'red', linewidth = 0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(myFmt)
    ax.get_xaxis().set_visible(True)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Log-ret')
    ax1.set_title('Forecast del prezzo ETH con i suoi futures')
    
    # Forecasting accuracy
    futures_regr_rmse = rmse(eth_log_ret[fin:fine], forecast)
    
    # Multipla
    s1 = pd.merge(eth_log_ret, np.log(NASDQ).diff().dropna(), how = 'inner', left_index = True, right_index = True)
    ETH_multipla = pd.merge(s1, np.log(VIX).diff().dropna(), how = 'inner', left_index = True, right_index = True)
    ETH_multipla.columns = ['ETH','NASDQ', 'VIX']
    ETH_multipla = ETH_multipla[inizio:fine]
    m_regr = ETH_multipla.loc[:, ['NASDQ', 'VIX']]
    m_regr.insert(0, 'constant', 1, allow_duplicates = True)
    # Dimensione della rolling window
    window = len(t_set(ETH_multipla, inizio, fine, fin, True))
    # Rolling window forecasting
    forecast = rolling_forecast_r(2, ETH_multipla.ETH, m_regr, window)
    # Plot
    fig, (ax1) = plt.subplots(1)
    ax1.plot(eth_log_ret[inizio:fine], label = 'ETH returns', color = 'grey', linewidth = 0.5)
    ax1.plot(date_forecast[86:], forecast, label = 'ETH forecast', color = 'pink', linewidth = 0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(myFmt)
    ax.get_xaxis().set_visible(True)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Log-ret')
    ax1.set_title('Forecast del prezzo ETH con NASDQ e VIX')
    
    # Forecasting accuracy
    simple_regr_rmse = rmse(eth_log_ret[fin:fine], forecast)
    
    # Indice
    #Log-Price Weighted Index
    import yfinance as yf
    # Lista dei simboli ticker
    tickers = [
        'BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD',
        'SOL-USD', 'XRP-USD', 'DOGE-USD', 'TON-USD',
        'ADA-USD', 'AVAX-USD'
    ]
    # Scaricamento dei dati per ciascun ticker
    data = {ticker: yf.download(ticker).dropna() for ticker in tickers}
    # Calcolo dell'indice
    log_price_index = calculate_weighted_log_price_index(data)
    log_price_index = pd.DataFrame(log_price_index)
    
    # Forecasting accuracy
    # Generalizziamo il forecasting in rolling window delle regressioni
    # con una funzione apposita
    ETH_index = pd.merge(eth_log_ret, np.log(log_price_index).diff().dropna(), how = 'inner', left_index = True, right_index = True)
    ETH_index.columns = ['ETH','INDEX']
    ETH_index = ETH_index[inizio:fine]
    index_regr = ETH_index.loc[:, ['INDEX']]
    index_regr.insert(0, 'constant', 1, allow_duplicates = True)
    # Dimensione della rolling window
    window = len(t_set(ETH_index, inizio, fine, fin, True))
    # Rolling window forecasting
    forecast = rolling_forecast_r(2, ETH_index.ETH, index_regr, window)
    # Plot
    myFmt = mdates.DateFormatter('%m-%y')
    date_forecast = dates[len(eth_log_ret[:fin])+1:len(eth_log_ret[:fine])]
    fig, (ax1) = plt.subplots(1)
    ax1.plot(eth_log_ret[inizio:fine], label = 'ETH returns', color = 'purple', linewidth = 0.5)
    ax1.plot(date_forecast, forecast, label = 'ETH forecast', color = 'green', linewidth = 0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(myFmt)
    ax.get_xaxis().set_visible(True)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Log-ret')
    ax1.set_title('Forecast del prezzo ETH con indice log-price-weighted')
    
    # Forecasting accuracy
    simple_regr_rmse = rmse(eth_log_ret[fin:fine], forecast)
    
    
    # Utilizzo di un modello ARMA
    # Studio di ACF e pACF su un campione ristretto
    start = '07-01-2023'
    finish = '31-12-2023'
    # ACF e pACF
    T = len(eth_log_ret[start:finish])
    # Due subplot, uno per ogni dato da rappresentare
    fig, axs = plt.subplots(1, 2, figsize = (16, 9))
    # Autocorrelation function con 50 lags
    acf_eth = sm.tsa.acf(eth_log_ret[start:finish], nlags = 50)[1:]
    Lag = range(1, 51)
    axs[0].stem(Lag, acf_eth, markerfmt = 'ro', linefmt = 'k-', basefmt = 'k--', label = "ACF")
    axs[0].axhline(y= 1.64 / np.sqrt(T), color = 'blue' , linestyle = '--', label = 'C.I.')
    axs[0].axhline(y= -1.64 / np.sqrt(T), color = 'blue', linestyle = '--', label = 'C.I.')
    axs[0].axis(ymin = -1, ymax = 1)
    axs[0].set_xlabel('Lag')
    axs[0].set_ylabel('')
    axs[0].set_title(r'ACF, ETH log-returns')
    axs[0].grid(True)
    # Partial autocorrelation function con 50 lags
    pacf_eth = sm.tsa.pacf(eth_log_ret[start:finish], nlags= 50)[1:]
    axs[1].stem(Lag, pacf_eth, markerfmt = 'ro', linefmt = 'k-', basefmt = 'k--', label = "pACF")
    axs[1].axhline(y = 1.64 / np.sqrt(T), color = 'blue', linestyle = '--', label = 'C.I.')
    axs[1].axhline(y = -1.64 / np.sqrt(T), color = 'blue', linestyle = '--', label = 'C.I.')
    axs[1].axis(ymin = -1,ymax = 1)
    axs[1].set_xlabel('Lag')
    axs[1].set_ylabel('')
    axs[1].set_title(r'pACF, ETH log-returns')
    axs[1].grid(True)
    #Show
    plt.tight_layout()
    plt.show()
    
    # Auto-ARIMA e scelta del lag ottimale
    # Seleziono automaticamente il miglior modello
    model = pm.auto_arima(eth_log_ret[start:], 
                          m=1,                                    
                          seasonal=False,  
                          d=None,             
                          test='adf',         # seleziono ADF
                          start_p=0, start_q=0,
                          max_p=15, max_q=15, 
                          D=None,            
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True, 
                          stepwise=True)
    print(model.summary())

    # Stima del modello appropriato in rolling window
    out = arma_forecast(start, finish, eth_log_ret, 5, 1)
    forecast, aic, bic, residuals2 = out[0], out[1], out[2], out[3]
    # Plot
    fig, (ax1) = plt.subplots(1)
    ax1.plot(eth_log_ret[start:], label = 'ETH returns', color = 'purple', linewidth = 0.5)
    ax1.plot(pd.date_range(start = '01-01-2024', end = '04-10-2024'), forecast, label = 'ETH forecast', color = 'green', linewidth = 0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(myFmt)
    ax.get_xaxis().set_visible(True)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Log-ret')
    ax1.set_title('Forecast del prezzo ETH con ARMA(5,1)')
    
    # Stima del modello appropriato in rolling window
    out2 = arma_forecast(start, finish, eth_log_ret, 1, 1)
    forecast2, aic2, bic2, residuals = out2[0], out2[1], out2[2], out2[3]
    # Plot
    fig, (ax1) = plt.subplots(1)
    ax1.plot(eth_log_ret[start:], label = 'ETH returns', color = 'purple', linewidth = 0.5)
    ax1.plot(pd.date_range(start = '01-01-2024', end = '04-10-2024'), forecast2, label = 'ETH forecast', color = 'green', linewidth = 0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(myFmt)
    ax.get_xaxis().set_visible(True)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Log-ret')
    ax1.set_title('Forecast del prezzo ETH con ARMA(1,1)')
    
    # AIC e BIC
    # Due subplot, uno per ogni dato da rappresentare
    fig, axs = plt.subplots(1, 2, figsize = (16, 9))
    axs[0].plot(aic)
    axs[0].plot(aic2)
    axs[0].set_xlabel('Rolling model')
    axs[0].set_ylabel('')
    axs[0].set_title(r'Akaike information criteria')
    axs[0].grid(True)
    axs[1].plot(bic)
    axs[1].plot(bic2)
    axs[1].set_xlabel('Rolling model')
    axs[1].set_ylabel('')
    axs[1].set_title(r'Bayesian information criteria')
    axs[1].grid(True)
    
    # Forecasting accuracy
    ARMA_rmse = rmse(eth_log_ret['01-01-2024':'04-10-2024'], forecast)
    ARMA_mape = mape_mae(eth_log_ret['01-01-2024':'04-10-2024'], forecast)[0]
    ARMA_mae = mape_mae(eth_log_ret['01-01-2024':'04-10-2024'], forecast)[1]
    # Forecasting accuracy
    ARMA_rmse1 = rmse(eth_log_ret['01-01-2024':'04-10-2024'], forecast2)
    ARMA_mape1 = mape_mae(eth_log_ret['01-01-2024':'04-10-2024'], forecast2)[0]
    ARMA_mae1 = mape_mae(eth_log_ret['01-01-2024':'04-10-2024'], forecast2)[1]
    
    # Residuals
    fig, (ax1) = plt.subplots(1)
    ax1.plot(residuals, label = 'Residuals', color = 'green', linewidth = 0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(myFmt)
    ax.get_xaxis().set_visible(False)
    ax1.set_xlabel('')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals del forecast ARMA(1,1) di ETH')
    
    mdl = ARIMA(eth_log_ret[start:], order = (1, 0, 1))
    stats_mdl = mdl.fit()
    stats_mdl.plot_diagnostics(figsize = (15, 10))
    plt.show()

    # White noise ARCH effect test sui residui dell'ARMA semi-ottimale
    print(sm.stats.acorr_ljungbox(residuals2, lags=[1], return_df=True))
    print(sm.stats.diagnostic.het_arch(residuals2, maxlag=None, store=False, ddof=2))
    
    # Utilizzo di GARCH models
    # Studio delle autocorrelazioni del momento secondo tramite ACF e pACFo
    # ACF e pACF
    T = len((eth_log_ret**2)[start:finish])
    # Tre subplot, uno per ogni dato da rappresentare
    fig, axs = plt.subplots(1, 3, figsize = (16, 9))
    # Rendimenti logaritmici della serie dei prezzi
    axs[0].plot((eth_log_ret**2)[start:finish], marker = 'o', linestyle = '-', color = 'cyan')
    axs[0].set_xlabel('Data')
    axs[0].set_ylabel('')
    axs[0].set_title(r'ETH squared log returns')
    axs[0].grid(True)
    # Autocorrelation function con 50 lags
    acf_eth = sm.tsa.acf((eth_log_ret**2)[start:finish], nlags = 50)[1:]
    Lag = range(1, 51)
    axs[1].stem(Lag, acf_eth, markerfmt = 'ro', linefmt = 'k-', basefmt = 'k--', label = "ACF")
    axs[1].axhline(y= 1.64 / np.sqrt(T), color = 'blue' , linestyle = '--', label = 'C.I.')
    axs[1].axhline(y= -1.64 / np.sqrt(T), color = 'blue', linestyle = '--', label = 'C.I.')
    axs[1].axis(ymin = -1, ymax = 1)
    axs[1].set_xlabel('Lag')
    axs[1].set_ylabel('')
    axs[1].set_title(r'ACF, ETH squared log-returns')
    axs[1].grid(True)
    # Partial autocorrelation function con 50 lags
    pacf_eth = sm.tsa.pacf((eth_log_ret**2)[start:finish], nlags= 50)[1:]
    axs[2].stem(Lag, pacf_eth, markerfmt = 'ro', linefmt = 'k-', basefmt = 'k--', label = "pACF")
    axs[2].axhline(y = 1.64 / np.sqrt(T), color = 'blue', linestyle = '--', label = 'C.I.')
    axs[2].axhline(y = -1.64 / np.sqrt(T), color = 'blue', linestyle = '--', label = 'C.I.')
    axs[2].axis(ymin = -1,ymax = 1)
    axs[2].set_xlabel('Lag')
    axs[2].set_ylabel('')
    axs[2].set_title(r'pACF, ETH squared log-returns')
    axs[2].grid(True)
    #Show
    plt.tight_layout()
    plt.show()
    
    #Stima del modello GARCH
    out = garch_forecast(start, finish, eth_log_ret, 1, 1);
    aic, bic, alpha, beta, omega, mu, forecast, returns = out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]
    # Sei subplot, uno per ogni dato da rappresentare
    fig, axs = plt.subplots(2, 3, figsize = (16, 9))
    axs[0,0].plot(aic, color = 'cyan', linewidth = 0.6)
    axs[0,0].set_title(r'Akaike information criteria')
    axs[0,0].grid(True)
    axs[0,1].plot(bic, color = 'red', linewidth = 0.6)
    axs[0,1].set_title(r'Bayesian information criteria')
    axs[0,1].grid(True)
    axs[0,2].plot(mu, color = 'blue', linewidth = 0.6)
    axs[0,2].set_xlabel('Model rolling')
    axs[0,2].set_ylabel('')
    axs[0,2].set_title(r'Mu of rolling garch')
    axs[0,2].grid(True)
    axs[1,0].plot(alpha, color = 'orange', linewidth = 0.6)
    axs[1,0].set_xlabel('Model rolling')
    axs[1,0].set_ylabel('')
    axs[1,0].set_title(r'Alpha of rolling garch')
    axs[1,0].grid(True)
    axs[1,1].plot(beta,  color = 'purple', linewidth = 0.6)
    axs[1,1].set_xlabel('Model rolling')
    axs[1,1].set_ylabel('')
    axs[1,1].set_title(r'Beta of rolling garch')
    axs[1,1].grid(True)
    axs[1,2].plot(omega, color = 'black', linewidth = 0.6)
    axs[1,2].set_xlabel('Model rolling')
    axs[1,2].set_ylabel('')
    axs[1,2].set_title(r'Omega of rolling garch')
    axs[1,2].grid(True)
    
    # Forecasting GARCH
    # Plot
    fig, (ax1) = plt.subplots(1)
    ax1.plot(eth_log_ret[start:]**2, label = 'Volatility', color = 'blue', linewidth = 0.5)
    ax1.plot(pd.date_range(start = '01-01-2024', end = '04-10-2024'), forecast, label = 'GARCH forecast', color = 'red', linewidth = 0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(myFmt)
    ax.get_xaxis().set_visible(True)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('')
    ax1.set_title('Forecast con GARCH(1,1) T-student')
    
    # Forecasting returns with GARCH
    # Plot
    fig, (ax1) = plt.subplots(1)
    ax1.plot(eth_log_ret[start:], label = 'ETH log-returns', color = 'blue', linewidth = 0.5)
    ax1.plot(pd.date_range(start = '01-01-2024', end = '04-10-2024'), returns, label = 'GARCH forecast', color = 'red', linewidth = 0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(myFmt)
    ax.get_xaxis().set_visible(True)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('')
    ax1.set_title('Forecast con GARCH(1,1) T-student (2023)')
    
    # Forecasting accuracy
    GARCH_rmse = rmse(eth_log_ret['01-01-2024':'04-10-2024'], returns)
    GARCH_mape = mape_mae(eth_log_ret['01-01-2024':'04-10-2024'], returns)[0]
    GARCH_mae = mape_mae(eth_log_ret['01-01-2024':'04-10-2024'], returns)[1]

    #Stima del modello GARCH per il 2020 (halving)
    inizio = '02-01-2020'
    fine = '08-01-2020'
    eth2020 = eth_log_ret[:'11-01-2020']
    out = garch_forecast(inizio, fine, eth2020, 1, 1);
    aic, bic, alpha, beta, omega, mu, forecast, returns = out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]
    
    # Forecasting
    # Forecasting GARCH
    # Plot
    fig, (ax1) = plt.subplots(1)
    ax1.plot(eth_log_ret['02-01-2020':'11-01-2020']**2, label = 'Volatility', color = 'blue', linewidth = 0.5)
    ax1.plot(pd.date_range(start = '08-02-2020', end = '11-01-2020'), forecast, label = 'GARCH forecast', color = 'red', linewidth = 0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(myFmt)
    ax.get_xaxis().set_visible(True)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('')
    ax1.set_title('Forecast con GARCH(1,1) T-student (2020)')
    
    # Forecasting returns with GARCH
    # Plot
    fig, (ax1) = plt.subplots(1)
    ax1.plot(eth_log_ret['02-01-2020':'11-01-2020'], label = 'ETH log-returns', color = 'blue', linewidth = 0.5)
    ax1.plot(pd.date_range(start = '08-02-2020', end = '11-01-2020'), returns, label = 'GARCH forecast', color = 'red', linewidth = 0.5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(myFmt)
    ax.get_xaxis().set_visible(True)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('')
    ax1.set_title('Forecast con GARCH(1,1) T-student (2020)')
    
    # Forecasting accuracy
    GARCH_rmse2020 = rmse(eth_log_ret['08-02-2020':'11-01-2020'], returns)
    GARCH_mape2020 = mape_mae(eth_log_ret['08-02-2020':'11-01-2020'], returns)[0]
    GARCH_mae2020 = mape_mae(eth_log_ret['08-02-2020':'11-01-2020'], returns)[1]
    
    # Stima di un Vector Autoregressive Model
    # Fitting->forecasting out of sample e misure di adattamento
    eth_log_ret = eth_log_ret[start:];
    btc_log_ret = np.log(library['BTC']['Ultimo prz']).diff().dropna();
    vix_log_ret = np.log(library['VIX']['Ultimo prz']).diff().dropna();
    nasdq_log_ret = np.log(library['NASDQ']['Ultimo prz']).diff().dropna();

    # Creiamo una matrice dei rendimenti log da dividere in test e training set
    merge1 = pd.merge(eth_log_ret, btc_log_ret, how = 'inner', left_index = True, right_index = True);
    merge2 = pd.merge(merge1, vix_log_ret, how = 'inner', left_index = True, right_index = True);
    merge2.columns = ['ETH','BTC','VIX'];
    lret = pd.merge(merge2, nasdq_log_ret, how = 'inner', left_index = True, right_index = True);
    lret.columns = ['ETH','BTC','VIX','NASDQ'];
    
    # Modelling and forecasting
    nobs = len(np.log(ETH).diff().dropna()['01-01-2024':])
    df_train, df_test = lret[0:-nobs], lret[-nobs:]
    model = VAR(df_train)
    model_fitted = model.fit()
    print(model_fitted.summary())
    # Plot
    model_fitted.plot_forecast(nobs)
    lag_order = model_fitted.k_ar
    z = model_fitted.forecast(y=df_train.values[-lag_order:],steps = nobs)
    df_forecast = pd.DataFrame(z, index=df_test.index[-nobs:], columns=df_test.columns + '_2d')
    print(df_forecast)

    # Tabella di confronto
    myTable = PrettyTable(["Modello", "RMSE", "MAE", "MAPE"]) 
    # Add rows 
    myTable.add_row(["ARMA(5,1)", ARMA_rmse, ARMA_mae, ARMA_mape])
    myTable.add_row(["ARMA(1,1)", ARMA_rmse1, ARMA_mae1, ARMA_mape1])
    myTable.add_row(["GARCH(1,1) T-student (2023)", GARCH_rmse, GARCH_mae, GARCH_mape])
    myTable.add_row(["GARCH(1,1) T-student (2020)", GARCH_rmse2020, GARCH_mae2020, GARCH_mape2020])
    print(myTable)

# Spazio riservato alle funzioni accessorie

# Definizione del training set/test set
def t_set(data, inizio, fine, finestra, training_set):
    if training_set == True:
        return data[inizio:finestra]
    else:
        return data[finestra:fine]
    
# Rolling forecast per regressioni lineari
def rolling_forecast_r(ncoeff, information_set, regr, window):
    inizio = '2020-01-01'
    fine = '2023-12-31'
    fin = '2023-04-01'
    test_set = len(t_set(information_set, inizio, fine, fin, False))
    coeffs = []
    for i in range(0, test_set - 1):
        # Updating information set
        training_set = information_set[i:window + i]
        regr_r = regr[i:window + i]
        # Finding coeff
        coeffs.append(np.linalg.inv(regr_r.transpose() @ regr_r) @ regr_r.transpose() @ training_set)  
    forecasts = np.diag(regr[window+1:len(regr)] @ np.transpose(coeffs))
    return np.array(forecasts)

# Log-price weighted index
def calculate_weighted_log_price_index(data):
    # Estrazione dei prezzi di chiusura in un unico DataFrame
    close_prices = pd.DataFrame({ticker: df['Close'] for ticker, df in data.items()})
    # Calcolo dei logaritmi dei prezzi di chiusura
    log_prices = np.log(close_prices)
    # Calcolo dei pesi per ogni criptovaluta
    total_log_prices = log_prices.sum(axis=1)
    weights = log_prices.divide(total_log_prices, axis=0)
    # Calcolo dell'indice log price weighted
    weighted_log_price_index = (log_prices * weights).sum(axis=1)
    return weighted_log_price_index

# ARMA rolling window model est.
def arma_forecast(start, finish, information_set, p, q):
    forecast = []
    aics = []
    bics = []
    residuals = []
    training_set = information_set[start:finish]
    window = len(training_set)
    information_set = information_set[start:]
    length = len(information_set)
    for i in range(0, length-window):
        t = information_set[i:window+i]
        mod = ARIMA(t, order = (p, 0, q))
        res = mod.fit()
        pred = res.forecast()
        aic = res.aic
        bic = res.bic
        resd = res.resid
        forecast.append(pred)
        aics.append(aic)
        bics.append(bic)
        residuals.append(resd[0])
    forecast = np.array(forecast)
    aics = np.array(aics)
    bics = np.array(bics)
    residuals = np.array(resd)
    out = [forecast, aics, bics, residuals]
    return out

# GARCH rolling window model est.
def garch_forecast(start, finish, information_set, a, b):
    aics = []
    bics = []
    alphas = []
    omegas = []
    betas = []
    mus = []
    forecasts = []
    forecastsr = []
    training_set = information_set[start:finish]
    window = len(training_set)
    information_set = information_set[start:]
    length = len(information_set)
    for i in range(0, length-window):
        t = information_set[i:window+i]
        model = arch_model(t, p=a, o=0, q=b, vol = 'Garch', dist="StudentsT")
        res = model.fit()
        aic = res.aic
        bic = res.bic
        mu = res.params.mu
        alpha = res.params['alpha[1]']
        beta = res.params['beta[1]']
        omega = res.params.omega
        forecast = float(res.forecast().variance.iloc[0])
        # La funzione a volte va in "time-out" e restituisce degli output "estremi" che vanno eliminati
        # pregiudicando di fatto la stima
        if forecast > 0.08:
            forecast = 0.08
        # Returns
        eps_t = np.sqrt(forecast)*np.random.standard_t(window-3)
        ret = eps_t + mu
        if ret < - 0.1:
            ret = -0.1
        aics.append(aic)
        bics.append(bic)
        alphas.append(alpha)
        betas.append(beta)
        omegas.append(omega)
        mus.append(mu)
        forecastsr.append(ret)
        forecasts.append(forecast)
    aics, bics = np.array(aics), np.array(bics)
    alphas, betas, omegas = np.array(alphas), np.array(betas), np.array(omegas)
    mus = np.array(mus)
    forecasts = np.array(forecasts);
    out = [aics, bics, alphas, betas, omegas, mus, forecasts, forecastsr]
    return out

# RMSE function
def rmse(test_set, forecast):
    return np.sqrt(((np.array(test_set) - forecast)**2).mean())

# MAPE function
def mape_mae(test_set, forecast):
    test_set = np.array(test_set)
    n = len(test_set)
    ae = []
    ape = []
    for i in range(0, n):
        e = abs(test_set[i]-forecast[i])
        pe = abs((test_set[i]-forecast[i])/test_set[i])
        ae.append(e)
        ape.append(pe)
    out = [(np.sum(ape)*100)/n, np.sum(ae)/n]
    return out

# Da https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/?utm_content=cmp-true
# Adfuller test
def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    


#
#
# END #
main();