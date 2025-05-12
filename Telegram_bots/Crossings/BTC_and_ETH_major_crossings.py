import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import os
import glob
import datetime as dt
import json

cg_token = os.getenv("CoinGecko_token")

def get_cg_mcap(coin_name, days='max'):
    headers = {
        "accept": "application/json",
        "x-cg-pro-api-key": cg_token
    }
    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_name}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url, headers=headers)
    data_shows = json.loads(response.text)
    data = data_shows['market_caps']

    date = []
    mcap = []
    for i in range(len(data)):
        date.append(dt.datetime.fromtimestamp(data[i][0] / 1000))
        mcap.append(data[i][1])

    df = pd.DataFrame(data=mcap, index=date, columns=[coin_name])

    # Convert index to date only (drop time)
    df.index = pd.to_datetime(df.index).date

    # Keep only the last entry per day
    df = df.groupby(df.index).last()

    return df

# Get the directory where the script is located
project_folder = os.path.dirname(os.path.abspath(__file__))

# Define the output folder inside the project directory
base_path = os.path.join(project_folder, "output_figures")

def clear_output_folder(base_path):
    """Deletes all files from the specified folder without removing the folder itself."""
    if not os.path.exists(base_path):
        print(f"Folder '{base_path}' does not exist.")
        return
    
    files = glob.glob(os.path.join(base_path, "*"))  # Get all files in the folder
    for file in files:
        try:
            os.remove(file)  # Delete file
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    print(f"All files in '{base_path}' have been deleted.")

def calculate_counts(df):
  positive_counts = (df > 0).sum()
  negative_counts = (df < 0).sum()
  return positive_counts, negative_counts

def plot_token_below_ma_yf_api(yf_ticker, num_ma_days):

    # Fetch data
    token = get_cg_mcap(yf_ticker, days='max')
    token[f"{num_ma_days}_day_MA"] = token[yf_ticker].rolling(window=num_ma_days).mean()

    dates = []
    for i in range(len(token) - 15):
        if token.iloc[i][yf_ticker] > token.iloc[i][f"{num_ma_days}_day_MA"]:
            if token.iloc[i+1][yf_ticker] < token.iloc[i+1][f"{num_ma_days}_day_MA"]:
                if (token.iloc[i+1:i+16][yf_ticker] < token.iloc[i+1:i+16][f"{num_ma_days}_day_MA"]).all():
                    dates.append(token.index[i+1])

    returns = []
    for date in dates:
        idx = token.index.get_loc(date)
        close_MCAP = token.iloc[idx][yf_ticker]

        if idx + 365 < len(token):
            returns.append({
                "date": date,
                "30_day_return": (token.iloc[idx+30][yf_ticker] - close_MCAP) / close_MCAP,
                "45_day_return": (token.iloc[idx+45][yf_ticker] - close_MCAP) / close_MCAP,
                "60_day_return": (token.iloc[idx+60][yf_ticker] - close_MCAP) / close_MCAP,
                "75_day_return": (token.iloc[idx+75][yf_ticker] - close_MCAP) / close_MCAP,
                "90_day_return": (token.iloc[idx+90][yf_ticker] - close_MCAP) / close_MCAP,
                "120_day_return": (token.iloc[idx+120][yf_ticker] - close_MCAP) / close_MCAP,
                "150_day_return": (token.iloc[idx+150][yf_ticker] - close_MCAP) / close_MCAP,
                "180_day_return": (token.iloc[idx+180][yf_ticker] - close_MCAP) / close_MCAP,
                "365_day_return": (token.iloc[idx+365][yf_ticker] - close_MCAP) / close_MCAP
            })

    returns = pd.DataFrame(returns)

    average_returns = {
        "30_day_return": returns["30_day_return"].mean(),
        "45_day_return": returns["45_day_return"].mean(),
        "60_day_return": returns["60_day_return"].mean(),
        "75_day_return": returns["75_day_return"].mean(),
        "90_day_return": returns["90_day_return"].mean(),
        "120_day_return": returns["120_day_return"].mean(),
        "150_day_return": returns["150_day_return"].mean(),
        "180_day_return": returns["180_day_return"].mean(),
        "365_day_return": returns["365_day_return"].mean()
    }

    average_returns = pd.Series(average_returns)

    # Plot data
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.plot(token[yf_ticker], label=yf_ticker, linewidth=3.5)
    ax.plot(token[f"{num_ma_days}_day_MA"], label=f"{num_ma_days}-day MA", linewidth=3.5)
    for date in dates:
        ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

    # Add legend, title, and labels
    ax.legend(fontsize=22)
    ax.set_title(f"{yf_ticker} MCAP and {num_ma_days}-day Moving Average", fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xlabel('Date', fontsize=30, fontweight='bold')
    ax.set_ylabel('MCAP', fontsize=30, fontweight='bold')

    # Save the plot
    file_name = f"{yf_ticker}_close_{num_ma_days}dma_plot.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    # Plotting
    fig, ax = plt.subplots(figsize=(22, 11))
    bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3, label='Average Returns')

    # Add values on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=16, color='black', weight='bold', xytext=(0, 10),
                    textcoords='offset points')

    # Increase axis labels and ticks font size and make them bold
    ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax.set_xlabel('Number of days to check', fontsize=20, weight='bold')
    ax.set_ylabel('Average Returns', fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days', '365 days'])
    # Increase font size of legend
    ax.legend(fontsize=20)

    # Increase title font size
    ax.set_title('Average Returns for Different Holding Durations', fontsize=20, fontweight='bold')
    # Rotate x-axis labels to normal
    plt.xticks(rotation=45)

    # Save plot to Google Drive
    file_name = f"{yf_ticker}_close_{num_ma_days}dma_average_returns.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    ret  =returns.drop(['date'], axis=1)

    df = ret
    positive_counts, negative_counts = calculate_counts(df)

    fig3, ax3 = plt.subplots(figsize=(28, 14))

    # Plot positive and negative returns bars
    bars1 = ax3.bar(np.arange(9), positive_counts, color='green',label='positive returns', align='center', edgecolor='black', linewidth=3)
    bars2 = ax3.bar(np.arange(9), negative_counts, color='red',label='negative returns', align='center', bottom=positive_counts, edgecolor='black', linewidth=3)

    # Add value of positive_counts/(positive_counts + negative_counts) on top of each bar
    for bar1, bar2 in zip(bars1, bars2):
        total_counts = bar1.get_height() + bar2.get_height()
        ratio = bar1.get_height()*100 / total_counts
        ax3.annotate(f"{ratio:.2f} % positive",
                    (bar1.get_x() + bar1.get_width() / 2., bar1.get_height()),
                    ha='center', va='bottom', fontsize=20, color='white', weight='bold', xytext=(0, 10),
                    textcoords='offset points', rotation=90)

    ax3.legend(loc='upper left', fontsize=20)
    ax3.set_title(f"Number of positive and negative returns after {yf_ticker} MCAP cross below {num_ma_days} days moving averages line", fontsize=20, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax3.set_xlabel("period", fontsize=20, weight='bold')
    ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
    # Rotate x-axis labels to normal
    plt.xticks(rotation=45)

    file_name = f"{yf_ticker}_close_{num_ma_days}dma_number_pos_neg_returns.png"
    save_path = os.path.join(base_path, file_name)
    plt.savefig(save_path)
    return token

def plot_token_above_ma_yf_api(yf_ticker, num_ma_days):

    # Fetch data
    token = get_cg_mcap(yf_ticker, days='max')
    token[f"{num_ma_days}_day_MA"] = token[yf_ticker].rolling(window=num_ma_days).mean()

    dates = []
    for i in range(len(token) - 15):
        if token.iloc[i][yf_ticker] < token.iloc[i][f"{num_ma_days}_day_MA"]:
            if token.iloc[i+1][yf_ticker] > token.iloc[i+1][f"{num_ma_days}_day_MA"]:
                if (token.iloc[i+1:i+16][yf_ticker] > token.iloc[i+1:i+16][f"{num_ma_days}_day_MA"]).all():
                    dates.append(token.index[i+1])

    returns = []
    for date in dates:
        idx = token.index.get_loc(date)
        close_MCAP = token.iloc[idx][yf_ticker]

        if idx + 365 < len(token):
            returns.append({
                "date": date,
                "30_day_return": (token.iloc[idx+30][yf_ticker] - close_MCAP) / close_MCAP,
                "45_day_return": (token.iloc[idx+45][yf_ticker] - close_MCAP) / close_MCAP,
                "60_day_return": (token.iloc[idx+60][yf_ticker] - close_MCAP) / close_MCAP,
                "75_day_return": (token.iloc[idx+75][yf_ticker] - close_MCAP) / close_MCAP,
                "90_day_return": (token.iloc[idx+90][yf_ticker] - close_MCAP) / close_MCAP,
                "120_day_return": (token.iloc[idx+120][yf_ticker] - close_MCAP) / close_MCAP,
                "150_day_return": (token.iloc[idx+150][yf_ticker] - close_MCAP) / close_MCAP,
                "180_day_return": (token.iloc[idx+180][yf_ticker] - close_MCAP) / close_MCAP,
                "365_day_return": (token.iloc[idx+365][yf_ticker] - close_MCAP) / close_MCAP
            })

    returns = pd.DataFrame(returns)

    average_returns = {
        "30_day_return": returns["30_day_return"].mean(),
        "45_day_return": returns["45_day_return"].mean(),
        "60_day_return": returns["60_day_return"].mean(),
        "75_day_return": returns["75_day_return"].mean(),
        "90_day_return": returns["90_day_return"].mean(),
        "120_day_return": returns["120_day_return"].mean(),
        "150_day_return": returns["150_day_return"].mean(),
        "180_day_return": returns["180_day_return"].mean(),
        "365_day_return": returns["365_day_return"].mean()
    }

    average_returns = pd.Series(average_returns)

    # Plot data
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.plot(token[yf_ticker], label=yf_ticker, linewidth=3.5)
    ax.plot(token[f"{num_ma_days}_day_MA"], label=f"{num_ma_days}-day MA", linewidth=3.5)
    for date in dates:
        ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

    # Add legend, title, and labels
    ax.legend(fontsize=22)
    ax.set_title(f"{yf_ticker} MCAP and {num_ma_days}-day Moving Average", fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xlabel('Date', fontsize=30, fontweight='bold')
    ax.set_ylabel('MCAP', fontsize=30, fontweight='bold')

    # Save the plot
    file_name = f"{num_ma_days}dma_{yf_ticker}_close_plot.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    # Plotting
    fig, ax = plt.subplots(figsize=(22, 11))
    bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3, label='Average Returns')

    # Add values on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=16, color='black', weight='bold', xytext=(0, 10),
                    textcoords='offset points')

    # Increase axis labels and ticks font size and make them bold
    ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax.set_xlabel('Number of days to check', fontsize=20, weight='bold')
    ax.set_ylabel('Average Returns', fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days', '365 days'])
    # Increase font size of legend
    ax.legend(fontsize=20)

    # Increase title font size
    ax.set_title('Average Returns for Different Holding Durations', fontsize=20, fontweight='bold')
    # Rotate x-axis labels to normal
    plt.xticks(rotation=45)

    # Save plot to Google Drive
    file_name = f"{num_ma_days}dma_{yf_ticker}_close_average_returns.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    ret  =returns.drop(['date'], axis=1)

    df = ret
    positive_counts, negative_counts = calculate_counts(df)

    fig3, ax3 = plt.subplots(figsize=(28, 14))

    # Plot positive and negative returns bars
    bars1 = ax3.bar(np.arange(9), positive_counts, color='green',label='positive returns', align='center', edgecolor='black', linewidth=3)
    bars2 = ax3.bar(np.arange(9), negative_counts, color='red',label='negative returns', align='center', bottom=positive_counts, edgecolor='black', linewidth=3)

    # Add value of positive_counts/(positive_counts + negative_counts) on top of each bar
    for bar1, bar2 in zip(bars1, bars2):
        total_counts = bar1.get_height() + bar2.get_height()
        ratio = bar1.get_height()*100 / total_counts
        ax3.annotate(f"{ratio:.2f} % positive",
                    (bar1.get_x() + bar1.get_width() / 2., bar1.get_height()),
                    ha='center', va='bottom', fontsize=20, color='white', weight='bold', xytext=(0, 10),
                    textcoords='offset points', rotation=90)

    ax3.legend(loc='upper left', fontsize=20)
    ax3.set_title(f"Number of positive and negative returns after {yf_ticker} MCAP cross above {num_ma_days} days moving averages line", fontsize=20, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax3.set_xlabel("period", fontsize=20, weight='bold')
    ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
    # Rotate x-axis labels to normal
    plt.xticks(rotation=45)

    file_name = f"{num_ma_days}dma_{yf_ticker}_close_number_pos_neg_returns.png"
    save_path = os.path.join(base_path, file_name)
    plt.savefig(save_path)
    return token

def plot_ratio_below_ma_yf_api(yf_ticker_a, yf_ticker_b, num_ma_days):
    # Fetch data for both tickers
    token_a = get_cg_mcap(yf_ticker_a, days='max')
    token_b = get_cg_mcap(yf_ticker_b, days='max')
    
    # Ensure both tokens have overlapping dates
    #token_b = token_b.tz_convert('UTC')
    
    # Convert index to date
    #token_a.index = token_a.index.date
    #token_b.index = token_b.index.date
    
    # Merge the dataframes on 'Date' column
    df = pd.merge(token_a, token_b, left_index=True, right_index=True, how='inner')
    # Calculate the ratio
    df['Ratio'] = df[yf_ticker_a] / df[yf_ticker_b]
    df[f"{num_ma_days}_day_MA"] = df["Ratio"].rolling(window=num_ma_days).mean()
    dates = []
    for i in range(len(df) - 15):
        if df.iloc[i]["Ratio"] > df.iloc[i][f"{num_ma_days}_day_MA"]:
            if df.iloc[i+1]["Ratio"] < df.iloc[i+1][f"{num_ma_days}_day_MA"]:
                if (df.iloc[i+1:i+16]["Ratio"] < df.iloc[i+1:i+16][f"{num_ma_days}_day_MA"]).all():
                    dates.append(df.index[i+1])

    returns = []
    for date in dates:
        idx = df.index.get_loc(date)
        close_MCAP = df.iloc[idx][yf_ticker_a]

        if idx + 365 < len(df):
            returns.append({
                "date": date,
                "30_day_return": (df.iloc[idx+30][yf_ticker_a] - close_MCAP) / close_MCAP,
                "45_day_return": (df.iloc[idx+45][yf_ticker_a] - close_MCAP) / close_MCAP,
                "60_day_return": (df.iloc[idx+60][yf_ticker_a] - close_MCAP) / close_MCAP,
                "75_day_return": (df.iloc[idx+75][yf_ticker_a] - close_MCAP) / close_MCAP,
                "90_day_return": (df.iloc[idx+90][yf_ticker_a] - close_MCAP) / close_MCAP,
                "120_day_return": (df.iloc[idx+120][yf_ticker_a] - close_MCAP) / close_MCAP,
                "150_day_return": (df.iloc[idx+150][yf_ticker_a] - close_MCAP) / close_MCAP,
                "180_day_return": (df.iloc[idx+180][yf_ticker_a] - close_MCAP) / close_MCAP,
                "365_day_return": (df.iloc[idx+365][yf_ticker_a] - close_MCAP) / close_MCAP
            })

    returns = pd.DataFrame(returns)

    average_returns = {
        "30_day_return": returns["30_day_return"].mean(),
        "45_day_return": returns["45_day_return"].mean(),
        "60_day_return": returns["60_day_return"].mean(),
        "75_day_return": returns["75_day_return"].mean(),
        "90_day_return": returns["90_day_return"].mean(),
        "120_day_return": returns["120_day_return"].mean(),
        "150_day_return": returns["150_day_return"].mean(),
        "180_day_return": returns["180_day_return"].mean(),
        "365_day_return": returns["365_day_return"].mean()
    }

    average_returns = pd.Series(average_returns)

    # Plot data
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.plot(df.index, df["Ratio"], label="Ratio", linewidth=3.5)
    ax.plot(df.index, df[f"{num_ma_days}_day_MA"], label=f"{num_ma_days}-day MA", linewidth=3.5)
    for date in dates:
        ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

    # Add legend, title, and labels
    ax.legend(fontsize=22)
    ax.set_title(f"{yf_ticker_a}/{yf_ticker_b} Ratio and {num_ma_days}-day Moving Average", fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xlabel('Date', fontsize=30, fontweight='bold')
    ax.set_ylabel('Ratio', fontsize=30, fontweight='bold')

    # Save the plot
    file_name = f"{yf_ticker_a}_{yf_ticker_b}_ratio_{num_ma_days}dma_plot.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # Plotting average returns
    fig, ax = plt.subplots(figsize=(22, 11))
    bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3, label='Average Returns')

    # Add values on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=16, color='black', weight='bold', xytext=(0, 10),
                    textcoords='offset points')

    # Increase axis labels and ticks font size and make them bold
    ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax.set_xlabel('Number of days to check', fontsize=20, weight='bold')
    ax.set_ylabel('Average Returns', fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days', '365 days'])
    ax.legend(fontsize=20)

    # Increase title font size
    ax.set_title('Average Returns for Different Holding Durations', fontsize=20, fontweight='bold')
    plt.xticks(rotation=45)

    # Save plot
    file_name = f"{num_ma_days}dma_{yf_ticker_a}_{yf_ticker_b}_ratio_{yf_ticker_a}_close_average_returns.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    ret = returns.drop(['date'], axis=1)
    positive_counts, negative_counts = calculate_counts(ret)

    fig3, ax3 = plt.subplots(figsize=(28, 14))

    # Plot positive and negative returns bars
    bars1 = ax3.bar(np.arange(9), positive_counts, color='green', label='positive returns', align='center', edgecolor='black', linewidth=3)
    bars2 = ax3.bar(np.arange(9), negative_counts, color='red', label='negative returns', align='center', bottom=positive_counts, edgecolor='black', linewidth=3)

    # Add value of positive_counts/(positive_counts + negative_counts) on top of each bar
    for bar1, bar2 in zip(bars1, bars2):
        total_counts = bar1.get_height() + bar2.get_height()
        ratio = bar1.get_height()*100 / total_counts
        ax3.annotate(f"{ratio:.2f} % positive",
                    (bar1.get_x() + bar1.get_width() / 2., bar1.get_height()),
                    ha='center', va='bottom', fontsize=20, color='white', weight='bold', xytext=(0, 10),
                    textcoords='offset points', rotation=90)

    ax3.legend(loc='upper left', fontsize=20)
    ax3.set_title(f"Number of positive and negative returns of {yf_ticker_a} after {yf_ticker_a}/{yf_ticker_b} ratio cross below {num_ma_days} days moving averages line", fontsize=20, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax3.set_xlabel("period", fontsize=20, weight='bold')
    ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
    plt.xticks(rotation=45)

    file_name = f"{num_ma_days}dma_{yf_ticker_a}_{yf_ticker_b}_ratio_{yf_ticker_a}_close_number_pos_neg_returns.png"
    save_path = os.path.join(base_path, file_name)
    plt.savefig(save_path)
    plt.close()
    
    return df

def plot_ratio_above_ma_yf_api(yf_ticker_a, yf_ticker_b, num_ma_days):
    # Fetch data for both tickers
    token_a = get_cg_mcap(yf_ticker_a, days='max')
    token_b = get_cg_mcap(yf_ticker_b, days='max')
    
    # Ensure both tokens have overlapping dates
    #token_b = token_b.tz_convert('UTC')
    
    # Convert index to date
    #token_a.index = token_a.index.date
    #token_b.index = token_b.index.date
    
    # Merge the dataframes on 'Date' column
    df = pd.merge(token_a, token_b, left_index=True, right_index=True, how='inner')
    
    # Calculate the ratio
    df['Ratio'] = df[yf_ticker_a] / df[yf_ticker_b]
    df[f"{num_ma_days}_day_MA"] = df["Ratio"].rolling(window=num_ma_days).mean()
    
    dates = []
    for i in range(len(df) - 15):
        if df.iloc[i]["Ratio"] < df.iloc[i][f"{num_ma_days}_day_MA"]:
            if df.iloc[i+1]["Ratio"] > df.iloc[i+1][f"{num_ma_days}_day_MA"]:
                if (df.iloc[i+1:i+16]["Ratio"] > df.iloc[i+1:i+16][f"{num_ma_days}_day_MA"]).all():
                    dates.append(df.index[i+1])

    returns = []
    for date in dates:
        idx = df.index.get_loc(date)
        close_MCAP = df.iloc[idx][yf_ticker_a]

        if idx + 365 < len(df):
            returns.append({
                "date": date,
                "30_day_return": (df.iloc[idx+30][yf_ticker_a] - close_MCAP) / close_MCAP,
                "45_day_return": (df.iloc[idx+45][yf_ticker_a] - close_MCAP) / close_MCAP,
                "60_day_return": (df.iloc[idx+60][yf_ticker_a] - close_MCAP) / close_MCAP,
                "75_day_return": (df.iloc[idx+75][yf_ticker_a] - close_MCAP) / close_MCAP,
                "90_day_return": (df.iloc[idx+90][yf_ticker_a] - close_MCAP) / close_MCAP,
                "120_day_return": (df.iloc[idx+120][yf_ticker_a] - close_MCAP) / close_MCAP,
                "150_day_return": (df.iloc[idx+150][yf_ticker_a] - close_MCAP) / close_MCAP,
                "180_day_return": (df.iloc[idx+180][yf_ticker_a] - close_MCAP) / close_MCAP,
                "365_day_return": (df.iloc[idx+365][yf_ticker_a] - close_MCAP) / close_MCAP
            })

    returns = pd.DataFrame(returns)

    average_returns = {
        "30_day_return": returns["30_day_return"].mean(),
        "45_day_return": returns["45_day_return"].mean(),
        "60_day_return": returns["60_day_return"].mean(),
        "75_day_return": returns["75_day_return"].mean(),
        "90_day_return": returns["90_day_return"].mean(),
        "120_day_return": returns["120_day_return"].mean(),
        "150_day_return": returns["150_day_return"].mean(),
        "180_day_return": returns["180_day_return"].mean(),
        "365_day_return": returns["365_day_return"].mean()
    }

    average_returns = pd.Series(average_returns)

    # Plot data
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.plot(df.index, df["Ratio"], label="Ratio", linewidth=3.5)
    ax.plot(df.index, df[f"{num_ma_days}_day_MA"], label=f"{num_ma_days}-day MA", linewidth=3.5)
    for date in dates:
        ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

    # Add legend, title, and labels
    ax.legend(fontsize=22)
    ax.set_title(f"{yf_ticker_a}/{yf_ticker_b} Ratio and {num_ma_days}-day Moving Average", fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xlabel('Date', fontsize=30, fontweight='bold')
    ax.set_ylabel('Ratio', fontsize=30, fontweight='bold')

    # Save the plot
    file_name = f"{yf_ticker_a}_{yf_ticker_b}_ratio_{num_ma_days}dma_plot_2.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # Plotting average returns
    fig, ax = plt.subplots(figsize=(22, 11))
    bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3, label='Average Returns')

    # Add values on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=16, color='black', weight='bold', xytext=(0, 10),
                    textcoords='offset points')

    # Increase axis labels and ticks font size and make them bold
    ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax.set_xlabel('Number of days to check', fontsize=20, weight='bold')
    ax.set_ylabel('Average Returns', fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days', '365 days'])
    ax.legend(fontsize=20)

    # Increase title font size
    ax.set_title('Average Returns for Different Holding Durations', fontsize=20, fontweight='bold')
    plt.xticks(rotation=45)

    # Save plot
    file_name = f"{num_ma_days}dma_{yf_ticker_a}_{yf_ticker_b}_ratio_{yf_ticker_a}_close_average_returns_2.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    ret = returns.drop(['date'], axis=1)
    positive_counts, negative_counts = calculate_counts(ret)

    fig3, ax3 = plt.subplots(figsize=(28, 14))

    # Plot positive and negative returns bars
    bars1 = ax3.bar(np.arange(9), positive_counts, color='green', label='positive returns', align='center', edgecolor='black', linewidth=3)
    bars2 = ax3.bar(np.arange(9), negative_counts, color='red', label='negative returns', align='center', bottom=positive_counts, edgecolor='black', linewidth=3)

    # Add value of positive_counts/(positive_counts + negative_counts) on top of each bar
    for bar1, bar2 in zip(bars1, bars2):
        total_counts = bar1.get_height() + bar2.get_height()
        ratio = bar1.get_height()*100 / total_counts
        ax3.annotate(f"{ratio:.2f} % positive",
                    (bar1.get_x() + bar1.get_width() / 2., bar1.get_height()),
                    ha='center', va='bottom', fontsize=20, color='white', weight='bold', xytext=(0, 10),
                    textcoords='offset points', rotation=90)

    ax3.legend(loc='upper left', fontsize=20)
    ax3.set_title(f"Number of positive and negative returns of {yf_ticker_a} after {yf_ticker_a}/{yf_ticker_b} ratio cross above {num_ma_days} days moving averages line", fontsize=20, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax3.set_xlabel("period", fontsize=20, weight='bold')
    ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
    plt.xticks(rotation=45)

    file_name = f"{num_ma_days}dma_{yf_ticker_a}_{yf_ticker_b}_ratio_{yf_ticker_a}_close_number_pos_neg_returns_2.png"
    save_path = os.path.join(base_path, file_name)
    plt.savefig(save_path)
    plt.close()
    
    return df

def plot_token_two_ma_yf_api(yf_ticker, num_ma_days_a, num_ma_days_b):
    """ num_ma_days_a goes below num_ma_days_b """

    # Fetch data
    token = get_cg_mcap(yf_ticker, days='max')
    token[f"{num_ma_days_a}_day_MA"] = token[yf_ticker].rolling(window=num_ma_days_a).mean()
    token[f"{num_ma_days_b}_day_MA"] = token[yf_ticker].rolling(window=num_ma_days_b).mean()

    dates = []
    for i in range(len(token) - 15):
        if token.iloc[i][f"{num_ma_days_a}_day_MA"] > token.iloc[i][f"{num_ma_days_b}_day_MA"]:
            if token.iloc[i+1][f"{num_ma_days_a}_day_MA"] < token.iloc[i+1][f"{num_ma_days_b}_day_MA"]:
                if (token.iloc[i+1:i+16][f"{num_ma_days_a}_day_MA"] < token.iloc[i+1:i+16][f"{num_ma_days_b}_day_MA"]).all():
                    dates.append(token.index[i+1])

    returns = []
    for date in dates:
        idx = token.index.get_loc(date)
        close_MCAP = token.iloc[idx][yf_ticker]

        if idx + 365 < len(token):
            returns.append({
                "date": date,
                "30_day_return": (token.iloc[idx+30][yf_ticker] - close_MCAP) / close_MCAP,
                "45_day_return": (token.iloc[idx+45][yf_ticker] - close_MCAP) / close_MCAP,
                "60_day_return": (token.iloc[idx+60][yf_ticker] - close_MCAP) / close_MCAP,
                "75_day_return": (token.iloc[idx+75][yf_ticker] - close_MCAP) / close_MCAP,
                "90_day_return": (token.iloc[idx+90][yf_ticker] - close_MCAP) / close_MCAP,
                "120_day_return": (token.iloc[idx+120][yf_ticker] - close_MCAP) / close_MCAP,
                "150_day_return": (token.iloc[idx+150][yf_ticker] - close_MCAP) / close_MCAP,
                "180_day_return": (token.iloc[idx+180][yf_ticker] - close_MCAP) / close_MCAP,
                "365_day_return": (token.iloc[idx+365][yf_ticker] - close_MCAP) / close_MCAP
            })

    returns = pd.DataFrame(returns)

    average_returns = {
        "30_day_return": returns["30_day_return"].mean(),
        "45_day_return": returns["45_day_return"].mean(),
        "60_day_return": returns["60_day_return"].mean(),
        "75_day_return": returns["75_day_return"].mean(),
        "90_day_return": returns["90_day_return"].mean(),
        "120_day_return": returns["120_day_return"].mean(),
        "150_day_return": returns["150_day_return"].mean(),
        "180_day_return": returns["180_day_return"].mean(),
        "365_day_return": returns["365_day_return"].mean()
    }

    average_returns = pd.Series(average_returns)

    # Plot data
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.plot(token[f"{num_ma_days_a}_day_MA"], label=f"{num_ma_days_a}_day_MA", linewidth=3.5)
    ax.plot(token[f"{num_ma_days_b}_day_MA"], label=f"{num_ma_days_b}-day MA", linewidth=3.5)
    for date in dates:
        ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

    # Add legend, title, and labels
    ax.legend(fontsize=22)
    ax.set_title(f"{yf_ticker} {num_ma_days_a} and {num_ma_days_b}-day Moving Average", fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xlabel('Date', fontsize=30, fontweight='bold')
    ax.set_ylabel('MCAP', fontsize=30, fontweight='bold')

    # Save the plot
    file_name = f"{yf_ticker}_{num_ma_days_a}_{num_ma_days_b}dma_plot.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    # Plotting
    fig, ax = plt.subplots(figsize=(22, 11))
    bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3, label='Average Returns')

    # Add values on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=16, color='black', weight='bold', xytext=(0, 10),
                    textcoords='offset points')

    # Increase axis labels and ticks font size and make them bold
    ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax.set_xlabel('Number of days to check', fontsize=20, weight='bold')
    ax.set_ylabel('Average Returns', fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days', '365 days'])
    # Increase font size of legend
    ax.legend(fontsize=20)

    # Increase title font size
    ax.set_title('Average Returns for Different Holding Durations', fontsize=20, fontweight='bold')
    # Rotate x-axis labels to normal
    plt.xticks(rotation=45)

    # Save plot to Google Drive
    file_name = f"{yf_ticker}_{num_ma_days_a}_{num_ma_days_b}dma_average_returns.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    ret  =returns.drop(['date'], axis=1)

    df = ret
    positive_counts, negative_counts = calculate_counts(df)

    fig3, ax3 = plt.subplots(figsize=(28, 14))

    # Plot positive and negative returns bars
    bars1 = ax3.bar(np.arange(9), positive_counts, color='green',label='positive returns', align='center', edgecolor='black', linewidth=3)
    bars2 = ax3.bar(np.arange(9), negative_counts, color='red',label='negative returns', align='center', bottom=positive_counts, edgecolor='black', linewidth=3)

    # Add value of positive_counts/(positive_counts + negative_counts) on top of each bar
    for bar1, bar2 in zip(bars1, bars2):
        total_counts = bar1.get_height() + bar2.get_height()
        ratio = bar1.get_height()*100 / total_counts
        ax3.annotate(f"{ratio:.2f} % positive",
                    (bar1.get_x() + bar1.get_width() / 2., bar1.get_height()),
                    ha='center', va='bottom', fontsize=20, color='white', weight='bold', xytext=(0, 10),
                    textcoords='offset points', rotation=90)

    ax3.legend(loc='upper left', fontsize=20)
    ax3.set_title(f"Number of positive and negative returns after {yf_ticker} {num_ma_days_a} cross below {num_ma_days_b} days moving averages line", fontsize=20, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax3.set_xlabel("period", fontsize=20, weight='bold')
    ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
    # Rotate x-axis labels to normal
    plt.xticks(rotation=45)

    file_name = f"{yf_ticker}_{num_ma_days_a}_{num_ma_days_b}dma_number_pos_neg_returns.png"
    save_path = os.path.join(base_path, file_name)
    plt.savefig(save_path)
    return token

def plot_ratio_two_ma_yf_api(yf_ticker_a, yf_ticker_b, num_ma_days_a, num_ma_days_b):
    """ num_ma_days_a goes below num_ma_days_b """

    # Fetch data for both tickers
    token_a = get_cg_mcap(yf_ticker_a, days='max')
    token_b = get_cg_mcap(yf_ticker_b, days='max')
    # Ensure both tokens have overlapping dates
    #token_b = token_b.tz_convert('UTC')
    # Convert index to date
    #token_a.index = token_a.index.date
    #token_b.index = token_b.index.date
    # Merge the dataframes on 'Date' column
    df = pd.merge(token_a, token_b, left_index=True, right_index=True, how='inner')

    # Now you can calculate the new column
    df['Ratio'] = df[yf_ticker_a] / df[yf_ticker_b]
    df[f"{num_ma_days_a}_day_MA"] = df["Ratio"].rolling(window=num_ma_days_a).mean()
    df[f"{num_ma_days_b}_day_MA"] = df["Ratio"].rolling(window=num_ma_days_b).mean()

    dates = []
    for i in range(len(df) - 15):
        if df.iloc[i][f"{num_ma_days_a}_day_MA"] > df.iloc[i][f"{num_ma_days_b}_day_MA"]:
            if df.iloc[i+1][f"{num_ma_days_a}_day_MA"] < df.iloc[i+1][f"{num_ma_days_b}_day_MA"]:
                if (df.iloc[i+1:i+16][f"{num_ma_days_a}_day_MA"] < df.iloc[i+1:i+16][f"{num_ma_days_b}_day_MA"]).all():
                    dates.append(df.index[i+1])

    returns = []
    for date in dates:
        idx = df.index.get_loc(date)
        close_MCAP = df.iloc[idx][yf_ticker_a]

        if idx + 365 < len(df):
            returns.append({
                "date": date,
                "30_day_return": (df.iloc[idx+30][yf_ticker_a] - close_MCAP) / close_MCAP,
                "45_day_return": (df.iloc[idx+45][yf_ticker_a] - close_MCAP) / close_MCAP,
                "60_day_return": (df.iloc[idx+60][yf_ticker_a] - close_MCAP) / close_MCAP,
                "75_day_return": (df.iloc[idx+75][yf_ticker_a] - close_MCAP) / close_MCAP,
                "90_day_return": (df.iloc[idx+90][yf_ticker_a] - close_MCAP) / close_MCAP,
                "120_day_return": (df.iloc[idx+120][yf_ticker_a] - close_MCAP) / close_MCAP,
                "150_day_return": (df.iloc[idx+150][yf_ticker_a] - close_MCAP) / close_MCAP,
                "180_day_return": (df.iloc[idx+180][yf_ticker_a] - close_MCAP) / close_MCAP,
                "365_day_return": (df.iloc[idx+365][yf_ticker_a] - close_MCAP) / close_MCAP
            })

    returns = pd.DataFrame(returns)

    average_returns = {
        "30_day_return": returns["30_day_return"].mean(),
        "45_day_return": returns["45_day_return"].mean(),
        "60_day_return": returns["60_day_return"].mean(),
        "75_day_return": returns["75_day_return"].mean(),
        "90_day_return": returns["90_day_return"].mean(),
        "120_day_return": returns["120_day_return"].mean(),
        "150_day_return": returns["150_day_return"].mean(),
        "180_day_return": returns["180_day_return"].mean(),
        "365_day_return": returns["365_day_return"].mean()
    }

    average_returns = pd.Series(average_returns)

    # Plot data
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.plot(df[f"{num_ma_days_a}_day_MA"], label=f"{num_ma_days_a}_day_MA", linewidth=3.5)
    ax.plot(df[f"{num_ma_days_b}_day_MA"], label=f"{num_ma_days_b}-day MA", linewidth=3.5)
    for date in dates:
        ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

    # Add legend, title, and labels
    ax.legend(fontsize=22)
    ax.set_title(f"{yf_ticker_a} {num_ma_days_a} and {num_ma_days_b}-day Moving Average", fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xlabel('Date', fontsize=30, fontweight='bold')
    ax.set_ylabel('MCAP', fontsize=30, fontweight='bold')

    # Save the plot
    file_name = f"{yf_ticker_a}_{num_ma_days_a}_{num_ma_days_b}dma_plot.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    # Plotting
    fig, ax = plt.subplots(figsize=(22, 11))
    bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3, label='Average Returns')

    # Add values on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=16, color='black', weight='bold', xytext=(0, 10),
                    textcoords='offset points')

    # Increase axis labels and ticks font size and make them bold
    ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax.set_xlabel('Number of days to check', fontsize=20, weight='bold')
    ax.set_ylabel('Average Returns', fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days', '365 days'])
    ax.legend(fontsize=20)

    # Increase title font size
    ax.set_title('Average Returns for Different Holding Durations', fontsize=20, fontweight='bold')
    plt.xticks(rotation=45)

    # Save plot
    file_name = f"{yf_ticker_a}_{num_ma_days_a}_{num_ma_days_b}dma_average_returns.png"
    save_path = os.path.join(base_path, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    ret  =returns.drop(['date'], axis=1)

    df = ret
    positive_counts, negative_counts = calculate_counts(df)

    fig3, ax3 = plt.subplots(figsize=(28, 14))

    # Plot positive and negative returns bars
    bars1 = ax3.bar(np.arange(9), positive_counts, color='green',label='positive returns', align='center', edgecolor='black', linewidth=3)
    bars2 = ax3.bar(np.arange(9), negative_counts, color='red',label='negative returns', align='center', bottom=positive_counts, edgecolor='black', linewidth=3)

    # Add value of positive_counts/(positive_counts + negative_counts) on top of each bar
    for bar1, bar2 in zip(bars1, bars2):
        total_counts = bar1.get_height() + bar2.get_height()
        ratio = bar1.get_height()*100 / total_counts
        ax3.annotate(f"{ratio:.2f} % positive",
                    (bar1.get_x() + bar1.get_width() / 2., bar1.get_height()),
                    ha='center', va='bottom', fontsize=20, color='white', weight='bold', xytext=(0, 10),
                    textcoords='offset points', rotation=90)

    ax3.legend(loc='upper left', fontsize=20)
    ax3.set_title(f"Number of positive and negative returns after {yf_ticker_a} {num_ma_days_a} cross below {num_ma_days_b} days moving averages line", fontsize=20, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
    ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
    ax3.set_xlabel("period", fontsize=20, weight='bold')
    ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
    plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
    # Rotate x-axis labels to normal
    plt.xticks(rotation=45)

    file_name = f"{yf_ticker_a}_{num_ma_days_a}_{num_ma_days_b}dma_number_pos_neg_returns.png"
    save_path = os.path.join(base_path, file_name)
    plt.savefig(save_path)
    return df

def send_telegram_message(token, chat_id, text):
    """Sends a text message to a Telegram chat."""
    try:
        url = f'https://api.telegram.org/bot{token}/sendMessage'
        response = requests.get(url, params={'chat_id': chat_id, 'text': text})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error sending Telegram message: {e}")

def send_telegram_photo(token, chat_id, caption, file_path):
    """Sends a photo with a caption to a Telegram chat."""
    try:
        url = f'https://api.telegram.org/bot{token}/sendPhoto'
        with open(file_path, 'rb') as photo:
            response = requests.post(url, data={'chat_id': chat_id, 'caption': caption}, files={'photo': photo})
        response.raise_for_status()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending Telegram photo: {e}")

def sending_to_tg_MCAP_below_ma(yf_ticker, num_ma_days, base_path):
    """Checks if the MCAP has crossed below the moving average and sends Telegram alerts."""
    token_df = plot_token_below_ma_yf_api(yf_ticker, num_ma_days)
    token = os.getenv("Tg_bot_token_crossings") 
    chat_id = '-4240308218'

    if token_df.iloc[-2][yf_ticker] > token_df.iloc[-2][f"{num_ma_days}_day_MA"] and \
       token_df.iloc[-1][yf_ticker] < token_df.iloc[-1][f"{num_ma_days}_day_MA"]:
        
        # Send text alert
        text = f'Attention, {yf_ticker} MCAP went below {num_ma_days}D MA!'
        send_telegram_message(token, chat_id, text)

        # Define images and captions
        image_info = [
            (f"{yf_ticker}_close_{num_ma_days}dma_plot.png", f'This is a timeseries chart of {yf_ticker} MCAP and {num_ma_days}D MA 😉'),
            (f"{yf_ticker}_close_{num_ma_days}dma_average_returns.png", f'These are the average returns after {yf_ticker} goes below {num_ma_days}D MA 😉'),
            (f"{yf_ticker}_close_{num_ma_days}dma_number_pos_neg_returns.png", f'This is the number of positive and negative returns after {yf_ticker} goes below {num_ma_days}D MA 😉')
        ]

        # Send each image
        for file_name, caption in image_info:
            file_path = os.path.join(base_path, file_name)
            send_telegram_photo(token, chat_id, caption, file_path)

def sending_to_tg_MCAP_above_ma(yf_ticker, num_ma_days, base_path):
    """Checks if the MCAP has crossed above the moving average and sends Telegram alerts."""
    token_df = plot_token_above_ma_yf_api(yf_ticker, num_ma_days)
    token = os.getenv("Tg_bot_token_crossings")
    chat_id = '-4240308218'

    if token_df.iloc[-2][yf_ticker] < token_df.iloc[-2][f"{num_ma_days}_day_MA"] and \
       token_df.iloc[-1][yf_ticker] > token_df.iloc[-1][f"{num_ma_days}_day_MA"]:
        
        # Send text alert
        text = f'Attention, {yf_ticker} MCAP went above {num_ma_days}D MA!'
        send_telegram_message(token, chat_id, text)

        # Define images and captions
        image_info = [
            (f"{num_ma_days}dma_{yf_ticker}_close_plot.png", f'This is a timeseries chart of {yf_ticker} MCAP and {num_ma_days}D MA 😉'),
            (f"{num_ma_days}dma_{yf_ticker}_close_average_returns.png", f'These are the average returns after {yf_ticker} goes above {num_ma_days}D MA 😉'),
            (f"{num_ma_days}dma_{yf_ticker}_close_number_pos_neg_returns.png", f'This is the number of positive and negative returns after {yf_ticker} goes above {num_ma_days}D MA 😉')
        ]

        # Send each image
        for file_name, caption in image_info:
            file_path = os.path.join(base_path, file_name)
            send_telegram_photo(token, chat_id, caption, file_path)

def sending_to_tg_ratio_below_ma(yf_ticker_a, yf_ticker_b, num_ma_days, base_path):
    """Checks if the MCAP has crossed below the moving average and sends Telegram alerts."""
    token_df = plot_ratio_below_ma_yf_api(yf_ticker_a, yf_ticker_b, num_ma_days)
    token = os.getenv("Tg_bot_token_crossings")
    chat_id = '-4240308218'

    if token_df.iloc[-2]["Ratio"] > token_df.iloc[-2][f"{num_ma_days}_day_MA"] and \
       token_df.iloc[-1]["Ratio"] < token_df.iloc[-1][f"{num_ma_days}_day_MA"]:
        
        # Send text alert
        text = f'Attention, {yf_ticker_a}/{yf_ticker_b} ratio went below {num_ma_days}D MA!'
        send_telegram_message(token, chat_id, text)

        # Define images and captions
        image_info = [
            (f"{yf_ticker_a}_{yf_ticker_b}_ratio_{num_ma_days}dma_plot.png", f'This is a timeseries chart of {yf_ticker_a}/{yf_ticker_b} ratio and {num_ma_days}D MA 😉'),
            (f"{num_ma_days}dma_{yf_ticker_a}_{yf_ticker_b}_ratio_{yf_ticker_a}_close_average_returns.png", f'These are the average returns of {yf_ticker_a} after {yf_ticker_a}/{yf_ticker_b} ratio goes below {num_ma_days}D MA 😉'),
            (f"{num_ma_days}dma_{yf_ticker_a}_{yf_ticker_b}_ratio_{yf_ticker_a}_close_number_pos_neg_returns.png", f'This is the number of positive and negative returns of {yf_ticker_a} after {yf_ticker_a}/{yf_ticker_b} ratio goes below {num_ma_days}D MA 😉')
        ]

        # Send each image
        for file_name, caption in image_info:
            file_path = os.path.join(base_path, file_name)
            send_telegram_photo(token, chat_id, caption, file_path)

def sending_to_tg_ratio_above_ma(yf_ticker_a, yf_ticker_b, num_ma_days, base_path):
    """Checks if the MCAP has crossed above the moving average and sends Telegram alerts."""
    token_df = plot_ratio_above_ma_yf_api(yf_ticker_a, yf_ticker_b, num_ma_days)
    token = os.getenv("Tg_bot_token_crossings")
    chat_id = '-4240308218'

    if token_df.iloc[-2]["Ratio"] < token_df.iloc[-2][f"{num_ma_days}_day_MA"] and \
       token_df.iloc[-1]["Ratio"] > token_df.iloc[-1][f"{num_ma_days}_day_MA"]:
        
        # Send text alert
        text = f'Attention, {yf_ticker_a}/{yf_ticker_b} ratio went above {num_ma_days}D MA!'
        send_telegram_message(token, chat_id, text)

        # Define images and captions
        image_info = [
            (f"{yf_ticker_a}_{yf_ticker_b}_ratio_{num_ma_days}dma_plot_2.png", f'This is a timeseries chart of {yf_ticker_a}/{yf_ticker_b} ratio and {num_ma_days}D MA 😉'),
            (f"{num_ma_days}dma_{yf_ticker_a}_{yf_ticker_b}_ratio_{yf_ticker_a}_close_average_returns_2.png", f'These are the average returns of {yf_ticker_a} after {yf_ticker_a}/{yf_ticker_b} ratio goes above {num_ma_days}D MA 😉'),
            (f"{num_ma_days}dma_{yf_ticker_a}_{yf_ticker_b}_ratio_{yf_ticker_a}_close_number_pos_neg_returns_2.png", f'This is the number of positive and negative returns of {yf_ticker_a} after {yf_ticker_a}/{yf_ticker_b} ratio goes above {num_ma_days}D MA 😉')
        ]

        # Send each image
        for file_name, caption in image_info:
            file_path = os.path.join(base_path, file_name)
            send_telegram_photo(token, chat_id, caption, file_path)

def sending_to_tg_two_ma(yf_ticker, num_ma_days_a, num_ma_days_b, base_path):
    """Checks if the num_ma_days_a has crossed below the num_ma_days_a moving average and sends Telegram alerts."""

    token_df = plot_token_two_ma_yf_api(yf_ticker, num_ma_days_a, num_ma_days_b)
    token = os.getenv("Tg_bot_token_crossings")
    chat_id = '-4240308218'

    if token_df.iloc[-2][f"{num_ma_days_a}_day_MA"] > token_df.iloc[-2][f"{num_ma_days_b}_day_MA"] and \
       token_df.iloc[-1][f"{num_ma_days_a}_day_MA"] < token_df.iloc[-1][f"{num_ma_days_b}_day_MA"]:
        
        # Send text alert
        text = f'Attention, {yf_ticker} {num_ma_days_a} went below {num_ma_days_b}D MA!'
        send_telegram_message(token, chat_id, text)

        # Define images and captions
        image_info = [
            (f"{yf_ticker}_{num_ma_days_a}_{num_ma_days_b}dma_plot.png", f'This is a timeseries chart of {yf_ticker} {num_ma_days_a} and {num_ma_days_b}D MA 😉'),
            (f"{yf_ticker}_{num_ma_days_a}_{num_ma_days_b}dma_average_returns.png", f'These are the average returns after {yf_ticker} {num_ma_days_a} goes below {num_ma_days_b}D MA 😉'),
            (f"{yf_ticker}_{num_ma_days_a}_{num_ma_days_b}dma_number_pos_neg_returns.png", f'This is the number of positive and negative returns after {yf_ticker} {num_ma_days_a} goes below {num_ma_days_b}D MA 😉')
        ]

        # Send each image
        for file_name, caption in image_info:
            file_path = os.path.join(base_path, file_name)
            send_telegram_photo(token, chat_id, caption, file_path)

def run_all_checks():
    """Run all specified crossing checks and send alerts if conditions are met."""
    print("Starting all crossing checks...")
    
    # List of MA periods to check
    ma_periods = [50, 100, 200, 365]
    
    try:
        # ethereum above MA checks
        print("\nChecking ethereum above MA...")
        for period in ma_periods:
            print(f"Checking ethereum above {period}D MA...")
            sending_to_tg_MCAP_above_ma("ethereum", period, base_path)
        
        # bitcoin above MA checks
        print("\nChecking bitcoin above MA...")
        for period in ma_periods:
            print(f"Checking bitcoin above {period}D MA...")
            sending_to_tg_MCAP_above_ma("bitcoin", period, base_path)
        
        # ethereum below MA checks
        print("\nChecking ethereum below MA...")
        for period in ma_periods:
            print(f"Checking ethereum below {period}D MA...")
            sending_to_tg_MCAP_below_ma("ethereum", period, base_path)
        
        # bitcoin below MA checks
        print("\nChecking bitcoin below MA...")
        for period in ma_periods:
            print(f"Checking bitcoin below {period}D MA...")
            sending_to_tg_MCAP_below_ma("bitcoin", period, base_path)
        
        # bitcoin and ^GSPC ratio checks
        #print("\nChecking bitcoin and ^GSPC ratios...")
        #sending_to_tg_ratio_below_ma("bitcoin", "^GSPC", 100, base_path)
        #sending_to_tg_ratio_above_ma("bitcoin", "^GSPC", 100, base_path)
        
        # bitcoin and ^GSPC MA ratio checks
        print("\nChecking bitcoin MA ratios...")
        sending_to_tg_two_ma("bitcoin", 50, 200, base_path)
        sending_to_tg_two_ma("bitcoin", 200, 50, base_path)
        
        # bitcoin and ethereum ratio checks
        print("\nChecking bitcoin ratios...")
        sending_to_tg_two_ma("bitcoin", 50, 200, base_path)
        sending_to_tg_two_ma("bitcoin", 200, 50, base_path)
        
        # ethereum MA ratio checks
        print("\nChecking ethereum MA ratios...")
        sending_to_tg_two_ma("ethereum", 50, 200, base_path)
        sending_to_tg_two_ma("ethereum", 200, 50, base_path)
        
        # bitcoin MA ratio checks
        print("\nChecking bitcoin MA ratios...")
        sending_to_tg_two_ma("bitcoin", 50, 200, base_path)
        sending_to_tg_two_ma("bitcoin", 200, 50, base_path)
        
        # Short-term MA ratio checks
        print("\nChecking short-term MA ratios...")
        # ethereum 7/30 MA
        sending_to_tg_two_ma("ethereum", 7, 30, base_path)
        sending_to_tg_two_ma("ethereum", 30, 7, base_path)
        # bitcoin 7/30 MA
        sending_to_tg_two_ma("bitcoin", 7, 30, base_path)
        sending_to_tg_two_ma("bitcoin", 30, 7, base_path)
        
        # bitcoin and ethereum ratio MA checks
        print("\nChecking bitcoin and ethereum ratio MAs...")
        # 50/200 MA
        sending_to_tg_two_ma("bitcoin", 50, 200, base_path)
        sending_to_tg_two_ma("bitcoin", 200, 50, base_path)
        # 7/30 MA
        sending_to_tg_two_ma("bitcoin", 7, 30, base_path)
        sending_to_tg_two_ma("bitcoin", 30, 7, base_path)
        
        # ethereum ratio MA checks
        print("\nChecking ethereum ratio MAs...")
        # 50/200 MA
        sending_to_tg_two_ma("ethereum", 50, 200, base_path)
        sending_to_tg_two_ma("ethereum", 200, 50, base_path)
        # 7/30 MA
        sending_to_tg_two_ma("ethereum", 7, 30, base_path)
        sending_to_tg_two_ma("ethereum", 30, 7, base_path)
        
        print("\n✅ All checks completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during checks: {str(e)}")
        return False

if __name__ == "__main__":
    # Clear output folder before running checks
    print("Clearing output folder...")
    clear_output_folder(base_path)
    
    # Run all checks
    run_all_checks()
    print("Clearing output folder...")
    clear_output_folder(base_path)
