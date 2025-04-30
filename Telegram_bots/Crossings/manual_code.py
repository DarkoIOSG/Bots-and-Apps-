import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

base_path = r"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings"

def calculate_counts(df):
  positive_counts = (df > 0).sum()
  negative_counts = (df < 0).sum()
  return positive_counts, negative_counts

########### ETH Price below 50DMA###############################

# Get Etheruem's data from yfinance library
eth = yf.Ticker("ETH-USD").history(period="max")

# Calculate 50, 100, and 200 day moving average
eth["50_day_MA"] = eth["Close"].rolling(window=50).mean()
eth["100_day_MA"] = eth["Close"].rolling(window=100).mean()
eth["200_day_MA"] = eth["Close"].rolling(window=200).mean()

dates = []
for i in range(len(eth) - 15):
    if eth.iloc[i]["Close"] > eth.iloc[i]["50_day_MA"]:
        if eth.iloc[i+1]["Close"] < eth.iloc[i+1]["50_day_MA"]:
            if (eth.iloc[i+1:i+16]["Close"] < eth.iloc[i+1:i+16]["50_day_MA"]).all():
                dates.append(eth.index[i+1])

returns = []
for date in dates:
    idx = eth.index.get_loc(date)
    close_price = eth.iloc[idx]["Close"]

    if idx + 365 < len(eth):
        returns.append({
            "date": date,
            "30_day_return": (eth.iloc[idx+30]["Close"] - close_price) / close_price,
            "45_day_return": (eth.iloc[idx+45]["Close"] - close_price) / close_price,
            "60_day_return": (eth.iloc[idx+60]["Close"] - close_price) / close_price,
            "75_day_return": (eth.iloc[idx+75]["Close"] - close_price) / close_price,
            "90_day_return": (eth.iloc[idx+90]["Close"] - close_price) / close_price,
            "120_day_return": (eth.iloc[idx+120]["Close"] - close_price) / close_price,
            "150_day_return": (eth.iloc[idx+150]["Close"] - close_price) / close_price,
            "180_day_return": (eth.iloc[idx+180]["Close"] - close_price) / close_price,
            "365_day_return": (eth.iloc[idx+365]["Close"] - close_price) / close_price
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
fig, ax = plt.subplots(figsize=(30,12))
ax.plot(eth["Close"], label="Close", linewidth = 3.5)
ax.plot(eth["50_day_MA"], label="50 day moving average", linewidth = 3.5)


# Add vertical lines for dates
for date in dates:
    ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

# Add legend and title
ax.legend(fontsize=22)
ax.set_title("Ethereum Price and Moving Averages", fontsize=28, fontweight = 'bold')
ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=22, width=2, length=3)
ax.set_xlabel('Date', fontsize=30, fontweight = 'bold')
ax.set_ylabel('Price', fontsize=30, fontweight = 'bold')

# Specify the exact path, and include the date and time in the filename
save_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_plot.png"
plt.savefig(save_path)

# Plotting
fig, ax = plt.subplots(figsize=(22, 11))
bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3)

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
save_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_average_returns.png"
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
ax3.set_title("Number of positive and negative returns after ETH price cross below 50 days moving averages line", fontsize=20, fontweight='bold')
ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
ax3.set_xlabel("period", fontsize=20, weight='bold')
ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
# Rotate x-axis labels to normal
plt.xticks(rotation=45)

# Save plot to Google Drive
save_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_number_pos_neg_returns.png"
plt.savefig(save_path)

if eth.iloc[len(eth)-2]["Close"] > eth.iloc[len(eth)-2]["50_day_MA"]:
   if eth.iloc[len(eth)-1]["Close"] < eth.iloc[len(eth)-1]["50_day_MA"]:
      token = 'X'
      msg_type = 'sendMessage'
      chat_id = '-4240308218'
      text = 'Attention, ETH price go below 50D MA!'
      msg = f'https://api.telegram.org/bot{token}/{msg_type}?chat_id={chat_id}&text={text}'
      telegram_msg = requests.get(msg)

      # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a timeseries chart ETH price and 50D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_plot.png\eth_close_50dma_plot.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))

        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This are a average returns after ETH goes below 50D MA 😉'
      local_img = rf'C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_plot.png\eth_close_50dma_average_returns.png' # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))


        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a number of positive and negative returns after ETH goes below 50D MA 😉'
      local_img = rf'C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_plot.png\eth_close_50dma_number_pos_neg_returns.png' # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))
      

########### ETH Price above 50DMA###############################

# Get Etheruem's data from yfinance library
eth = yf.Ticker("ETH-USD").history(period="max")

# Calculate 50, 100, and 200 day moving average
eth["50_day_MA"] = eth["Close"].rolling(window=50).mean()
eth["100_day_MA"] = eth["Close"].rolling(window=100).mean()
eth["200_day_MA"] = eth["Close"].rolling(window=200).mean()

dates = []
for i in range(len(eth) - 15):
    if eth.iloc[i]["Close"] < eth.iloc[i]["50_day_MA"]:
        if eth.iloc[i+1]["Close"] > eth.iloc[i+1]["50_day_MA"]:
            if (eth.iloc[i+1:i+16]["Close"] > eth.iloc[i+1:i+16]["50_day_MA"]).all():
                dates.append(eth.index[i+1])

returns = []
for date in dates:
    idx = eth.index.get_loc(date)
    close_price = eth.iloc[idx]["Close"]

    if idx + 365 < len(eth):
        returns.append({
            "date": date,
            "30_day_return": (eth.iloc[idx+30]["Close"] - close_price) / close_price,
            "45_day_return": (eth.iloc[idx+45]["Close"] - close_price) / close_price,
            "60_day_return": (eth.iloc[idx+60]["Close"] - close_price) / close_price,
            "75_day_return": (eth.iloc[idx+75]["Close"] - close_price) / close_price,
            "90_day_return": (eth.iloc[idx+90]["Close"] - close_price) / close_price,
            "120_day_return": (eth.iloc[idx+120]["Close"] - close_price) / close_price,
            "150_day_return": (eth.iloc[idx+150]["Close"] - close_price) / close_price,
            "180_day_return": (eth.iloc[idx+180]["Close"] - close_price) / close_price,
            "365_day_return": (eth.iloc[idx+365]["Close"] - close_price) / close_price
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
fig, ax = plt.subplots(figsize=(30,12))
ax.plot(eth["Close"], label="Close", linewidth = 3.5)
ax.plot(eth["50_day_MA"], label="50 day moving average", linewidth = 3.5)

# Add vertical lines for dates
for date in dates:
    ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

# Add legend and title
ax.legend(fontsize=22)
ax.set_title("Ethereum Price and 50D Moving Averages", fontsize=28, fontweight = 'bold')
ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=22, width=2, length=3)
ax.set_xlabel('Date', fontsize=30, fontweight = 'bold')
ax.set_ylabel('Price', fontsize=30, fontweight = 'bold')

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_plot2.png"
plt.savefig(file_path)

# Plotting
fig, ax = plt.subplots(figsize=(22, 11))
bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3)

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
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_average_returns2.png"
plt.savefig(file_path)

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
ax3.set_title("Number of positive and negative returns after ETH price cross above 50 days moving averages line", fontsize=20, fontweight='bold')
ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
ax3.set_xlabel("period", fontsize=20, weight='bold')
ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
# Rotate x-axis labels to normal
plt.xticks(rotation=45)

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_number_pos_neg_returns2.png"
plt.savefig(file_path)

if eth.iloc[len(eth)-2]["Close"] < eth.iloc[len(eth)-2]["50_day_MA"]:
   if eth.iloc[len(eth)-1]["Close"] > eth.iloc[len(eth)-1]["50_day_MA"]:
      token = 'X'
      msg_type = 'sendMessage'
      chat_id = '-4240308218'
      text = 'Attention, ETH price go above 50D MA!'
      msg = f'https://api.telegram.org/bot{token}/{msg_type}?chat_id={chat_id}&text={text}'
      telegram_msg = requests.get(msg)

      # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a timeseries chart ETH price and 50D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_plot2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))

        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This are a average returns after ETH goes above 50D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_average_returns2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))


        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a number of positive and negative returns after ETH goes above 50D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_50dma_number_pos_neg_returns2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))


########### ETH Price below 100DMA###############################

# Get Bitcoin's data from yfinance library
eth = yf.Ticker("ETH-USD").history(period="max")

# Calculate 50, 100, and 200 day moving average
eth["50_day_MA"] = eth["Close"].rolling(window=50).mean()
eth["100_day_MA"] = eth["Close"].rolling(window=100).mean()
eth["200_day_MA"] = eth["Close"].rolling(window=200).mean()

dates = []
for i in range(len(eth) - 15):
    if eth.iloc[i]["Close"] > eth.iloc[i]["100_day_MA"]:
        if eth.iloc[i+1]["Close"] < eth.iloc[i+1]["100_day_MA"]:
            if (eth.iloc[i+1:i+16]["Close"] < eth.iloc[i+1:i+16]["100_day_MA"]).all():
                dates.append(eth.index[i+1])

returns = []
for date in dates:
    idx = eth.index.get_loc(date)
    close_price = eth.iloc[idx]["Close"]

    if idx + 365 < len(eth):
        returns.append({
            "date": date,
            "30_day_return": (eth.iloc[idx+30]["Close"] - close_price) / close_price,
            "45_day_return": (eth.iloc[idx+45]["Close"] - close_price) / close_price,
            "60_day_return": (eth.iloc[idx+60]["Close"] - close_price) / close_price,
            "75_day_return": (eth.iloc[idx+75]["Close"] - close_price) / close_price,
            "90_day_return": (eth.iloc[idx+90]["Close"] - close_price) / close_price,
            "120_day_return": (eth.iloc[idx+120]["Close"] - close_price) / close_price,
            "150_day_return": (eth.iloc[idx+150]["Close"] - close_price) / close_price,
            "180_day_return": (eth.iloc[idx+180]["Close"] - close_price) / close_price,
            "365_day_return": (eth.iloc[idx+365]["Close"] - close_price) / close_price
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
fig, ax = plt.subplots(figsize=(30,12))
ax.plot(eth["Close"], label="Close", linewidth = 3.5)
ax.plot(eth["100_day_MA"], label="100 day moving average", linewidth = 3.5)

# Add vertical lines for dates
for date in dates:
    ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

# Add legend and title
ax.legend(fontsize=22)
ax.set_title("Ethereum Price and 100D Moving Averages", fontsize=28, fontweight = 'bold')
ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=22, width=2, length=3)
ax.set_xlabel('Date', fontsize=30, fontweight = 'bold')
ax.set_ylabel('Price', fontsize=30, fontweight = 'bold')

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_plot.png"
plt.savefig(file_path)

# Plotting
fig, ax = plt.subplots(figsize=(22, 11))
bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3)

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
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_average_returns.png"
plt.savefig(file_path)

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
ax3.set_title("Number of positive and negative returns after ETH price cross below 100 days moving averages line", fontsize=20, fontweight='bold')
ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
ax3.set_xlabel("period", fontsize=20, weight='bold')
ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
# Rotate x-axis labels to normal
plt.xticks(rotation=45)

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_number_pos_neg_returns.png"
plt.savefig(file_path)

if eth.iloc[len(eth)-2]["Close"] > eth.iloc[len(eth)-2]["100_day_MA"]:
   if eth.iloc[len(eth)-1]["Close"] < eth.iloc[len(eth)-1]["100_day_MA"]:
      token = 'X'
      msg_type = 'sendMessage'
      chat_id = '-4240308218'
      text = 'Attention, ETH price go below 100D MA!'
      msg = f'https://api.telegram.org/bot{token}/{msg_type}?chat_id={chat_id}&text={text}'
      telegram_msg = requests.get(msg)

      # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a timeseries chart ETH price and 100D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_plot.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))

        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This are a average returns after ETH goes below 100D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_average_returns.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))


        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a number of positive and negative returns after ETH goes below 100D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_number_pos_neg_returns.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))
      
########### ETH Price above 100DMA###############################

# Get Bitcoin's data from yfinance library
eth = yf.Ticker("ETH-USD").history(period="max")

# Calculate 50, 100, and 200 day moving average
eth["50_day_MA"] = eth["Close"].rolling(window=50).mean()
eth["100_day_MA"] = eth["Close"].rolling(window=100).mean()
eth["200_day_MA"] = eth["Close"].rolling(window=200).mean()

dates = []
for i in range(len(eth) - 15):
    if eth.iloc[i]["Close"] < eth.iloc[i]["100_day_MA"]:
        if eth.iloc[i+1]["Close"] > eth.iloc[i+1]["100_day_MA"]:
            if (eth.iloc[i+1:i+16]["Close"] > eth.iloc[i+1:i+16]["100_day_MA"]).all():
                dates.append(eth.index[i+1])

returns = []
for date in dates:
    idx = eth.index.get_loc(date)
    close_price = eth.iloc[idx]["Close"]

    if idx + 365 < len(eth):
        returns.append({
            "date": date,
            "30_day_return": (eth.iloc[idx+30]["Close"] - close_price) / close_price,
            "45_day_return": (eth.iloc[idx+45]["Close"] - close_price) / close_price,
            "60_day_return": (eth.iloc[idx+60]["Close"] - close_price) / close_price,
            "75_day_return": (eth.iloc[idx+75]["Close"] - close_price) / close_price,
            "90_day_return": (eth.iloc[idx+90]["Close"] - close_price) / close_price,
            "120_day_return": (eth.iloc[idx+120]["Close"] - close_price) / close_price,
            "150_day_return": (eth.iloc[idx+150]["Close"] - close_price) / close_price,
            "180_day_return": (eth.iloc[idx+180]["Close"] - close_price) / close_price,
            "365_day_return": (eth.iloc[idx+365]["Close"] - close_price) / close_price
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
fig, ax = plt.subplots(figsize=(30,12))
ax.plot(eth["Close"], label="Close", linewidth = 3.5)
ax.plot(eth["100_day_MA"], label="100 day moving average", linewidth = 3.5)

# Add vertical lines for dates
for date in dates:
    ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

# Add legend and title
ax.legend(fontsize=22)
ax.set_title("Ethereum Price and 100D Moving Averages", fontsize=28, fontweight = 'bold')
ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=22, width=2, length=3)
ax.set_xlabel('Date', fontsize=30, fontweight = 'bold')
ax.set_ylabel('Price', fontsize=30, fontweight = 'bold')

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_plot2.png"
plt.savefig(file_path)

# Plotting
fig, ax = plt.subplots(figsize=(22, 11))
bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3)

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
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_average_returns2.png"
plt.savefig(file_path)

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
ax3.set_title("Number of positive and negative returns after ETH price cross above 100 days moving averages line", fontsize=20, fontweight='bold')
ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
ax3.set_xlabel("period", fontsize=20, weight='bold')
ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
# Rotate x-axis labels to normal
plt.xticks(rotation=45)

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_number_pos_neg_returns2.png"
plt.savefig(file_path)

if eth.iloc[len(eth)-2]["Close"] < eth.iloc[len(eth)-2]["100_day_MA"]:
   if eth.iloc[len(eth)-1]["Close"] > eth.iloc[len(eth)-1]["100_day_MA"]:
      token = 'X'
      msg_type = 'sendMessage'
      chat_id = '-4240308218'
      text = 'Attention, ETH price go above 100D MA!'
      msg = f'https://api.telegram.org/bot{token}/{msg_type}?chat_id={chat_id}&text={text}'
      telegram_msg = requests.get(msg)

      # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a timeseries chart ETH price and 100D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_plot2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))

            # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This are a average returns after ETH goes above 100D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_average_returns2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))


        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a number of positive and negative returns after ETH goes above 100D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_100dma_number_pos_neg_returns2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))
      
########### ETH Price below 200DMA###############################

# Get Bitcoin's data from yfinance library
eth = yf.Ticker("ETH-USD").history(period="max")

# Calculate 50, 100, and 200 day moving average
eth["50_day_MA"] = eth["Close"].rolling(window=50).mean()
eth["100_day_MA"] = eth["Close"].rolling(window=100).mean()
eth["200_day_MA"] = eth["Close"].rolling(window=200).mean()

dates = []
for i in range(len(eth) - 15):
    if eth.iloc[i]["Close"] > eth.iloc[i]["200_day_MA"]:
        if eth.iloc[i+1]["Close"] < eth.iloc[i+1]["200_day_MA"]:
            if (eth.iloc[i+1:i+16]["Close"] < eth.iloc[i+1:i+16]["200_day_MA"]).all():
                dates.append(eth.index[i+1])

returns = []
for date in dates:
    idx = eth.index.get_loc(date)
    close_price = eth.iloc[idx]["Close"]

    if idx + 365 < len(eth):
        returns.append({
            "date": date,
            "30_day_return": (eth.iloc[idx+30]["Close"] - close_price) / close_price,
            "45_day_return": (eth.iloc[idx+45]["Close"] - close_price) / close_price,
            "60_day_return": (eth.iloc[idx+60]["Close"] - close_price) / close_price,
            "75_day_return": (eth.iloc[idx+75]["Close"] - close_price) / close_price,
            "90_day_return": (eth.iloc[idx+90]["Close"] - close_price) / close_price,
            "120_day_return": (eth.iloc[idx+120]["Close"] - close_price) / close_price,
            "150_day_return": (eth.iloc[idx+150]["Close"] - close_price) / close_price,
            "180_day_return": (eth.iloc[idx+180]["Close"] - close_price) / close_price,
            "365_day_return": (eth.iloc[idx+365]["Close"] - close_price) / close_price
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
fig, ax = plt.subplots(figsize=(30,12))
ax.plot(eth["Close"], label="Close", linewidth = 3.5)
ax.plot(eth["200_day_MA"], label="200 day moving average", linewidth = 3.5)

# Add vertical lines for dates
for date in dates:
    ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

# Add legend and title
ax.legend(fontsize=22)
ax.set_title("Ethereum Price and 200D Moving Averages", fontsize=28, fontweight = 'bold')
ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=22, width=2, length=3)
ax.set_xlabel('Date', fontsize=30, fontweight = 'bold')
ax.set_ylabel('Price', fontsize=30, fontweight = 'bold')

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_plot.png"
plt.savefig(file_path)

# Plotting
fig, ax = plt.subplots(figsize=(22, 11))
bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3)

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
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_average_returns.png"
plt.savefig(file_path)

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
ax3.set_title("Number of positive and negative returns after ETH price cross below 200 days moving averages line", fontsize=20, fontweight='bold')
ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
ax3.set_xlabel("period", fontsize=20, weight='bold')
ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
# Rotate x-axis labels to normal
plt.xticks(rotation=45)

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_number_pos_neg_returns.png"
plt.savefig(file_path)

if eth.iloc[len(eth)-2]["Close"] > eth.iloc[len(eth)-2]["200_day_MA"]:
   if eth.iloc[len(eth)-1]["Close"] < eth.iloc[len(eth)-1]["200_day_MA"]:
      token = 'X'
      msg_type = 'sendMessage'
      chat_id = '-4240308218'
      text = 'Attention, ETH price go below 200D MA!'
      msg = f'https://api.telegram.org/bot{token}/{msg_type}?chat_id={chat_id}&text={text}'
      telegram_msg = requests.get(msg)

      # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a timeseries chart ETH price and 200D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_plot.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))

        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This are a average returns after ETH goes below 200D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_average_returns.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))


        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a number of positive and negative returns after ETH goes below 200D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_number_pos_neg_returns.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))
      
########### ETH Price above 200DMA###############################

# Get Bitcoin's data from yfinance library
eth = yf.Ticker("ETH-USD").history(period="max")

# Calculate 50, 100, and 200 day moving average
eth["50_day_MA"] = eth["Close"].rolling(window=50).mean()
eth["100_day_MA"] = eth["Close"].rolling(window=100).mean()
eth["200_day_MA"] = eth["Close"].rolling(window=200).mean()

dates = []
for i in range(len(eth) - 15):
    if eth.iloc[i]["Close"] < eth.iloc[i]["200_day_MA"]:
        if eth.iloc[i+1]["Close"] > eth.iloc[i+1]["200_day_MA"]:
            if (eth.iloc[i+1:i+16]["Close"] > eth.iloc[i+1:i+16]["200_day_MA"]).all():
                dates.append(eth.index[i+1])

returns = []
for date in dates:
    idx = eth.index.get_loc(date)
    close_price = eth.iloc[idx]["Close"]

    if idx + 365 < len(eth):
        returns.append({
            "date": date,
            "30_day_return": (eth.iloc[idx+30]["Close"] - close_price) / close_price,
            "45_day_return": (eth.iloc[idx+45]["Close"] - close_price) / close_price,
            "60_day_return": (eth.iloc[idx+60]["Close"] - close_price) / close_price,
            "75_day_return": (eth.iloc[idx+75]["Close"] - close_price) / close_price,
            "90_day_return": (eth.iloc[idx+90]["Close"] - close_price) / close_price,
            "120_day_return": (eth.iloc[idx+120]["Close"] - close_price) / close_price,
            "150_day_return": (eth.iloc[idx+150]["Close"] - close_price) / close_price,
            "180_day_return": (eth.iloc[idx+180]["Close"] - close_price) / close_price,
            "365_day_return": (eth.iloc[idx+365]["Close"] - close_price) / close_price
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
fig, ax = plt.subplots(figsize=(30,12))
ax.plot(eth["Close"], label="Close", linewidth = 3.5)
ax.plot(eth["200_day_MA"], label="200 day moving average", linewidth = 3.5)

# Add vertical lines for dates
for date in dates:
    ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

# Add legend and title
ax.legend(fontsize=22)
ax.set_title("Ethereum Price and 200D Moving Averages", fontsize=28, fontweight = 'bold')
ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=22, width=2, length=3)
ax.set_xlabel('Date', fontsize=30, fontweight = 'bold')
ax.set_ylabel('Price', fontsize=30, fontweight = 'bold')

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_plot2.png"
plt.savefig(file_path)

# Plotting
fig, ax = plt.subplots(figsize=(22, 11))
bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3)

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
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_average_returns2.png"
plt.savefig(file_path)

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
ax3.set_title("Number of positive and negative returns after ETH price cross above 200 days moving averages line", fontsize=20, fontweight='bold')
ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
ax3.set_xlabel("period", fontsize=20, weight='bold')
ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
# Rotate x-axis labels to normal
plt.xticks(rotation=45)

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_number_pos_neg_returns2.png"
plt.savefig(file_path)

if eth.iloc[len(eth)-2]["Close"] < eth.iloc[len(eth)-2]["200_day_MA"]:
   if eth.iloc[len(eth)-1]["Close"] > eth.iloc[len(eth)-1]["200_day_MA"]:
      token = 'X'
      msg_type = 'sendMessage'
      chat_id = '-4240308218'
      text = 'Attention, ETH price go above 200D MA!'
      msg = f'https://api.telegram.org/bot{token}/{msg_type}?chat_id={chat_id}&text={text}'
      telegram_msg = requests.get(msg)

      # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a timeseries chart ETH price and 200D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_plot2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))

        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This are a average returns after ETH goes above 200D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_average_returns2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))


        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a number of positive and negative returns after ETH goes above 200D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_200dma_number_pos_neg_returns2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))
      
########### ETH Price below 365DMA###############################

# Get Bitcoin's data from yfinance library
eth = yf.Ticker("ETH-USD").history(period="max")

# Calculate 50, 100, and 200 day moving average
eth["50_day_MA"] = eth["Close"].rolling(window=50).mean()
eth["100_day_MA"] = eth["Close"].rolling(window=100).mean()
eth["365_day_MA"] = eth["Close"].rolling(window=365).mean()

dates = []
for i in range(len(eth) - 15):
    if eth.iloc[i]["Close"] > eth.iloc[i]["365_day_MA"]:
        if eth.iloc[i+1]["Close"] < eth.iloc[i+1]["365_day_MA"]:
            if (eth.iloc[i+1:i+16]["Close"] < eth.iloc[i+1:i+16]["365_day_MA"]).all():
                dates.append(eth.index[i+1])

returns = []
for date in dates:
    idx = eth.index.get_loc(date)
    close_price = eth.iloc[idx]["Close"]

    if idx + 365 < len(eth):
        returns.append({
            "date": date,
            "30_day_return": (eth.iloc[idx+30]["Close"] - close_price) / close_price,
            "45_day_return": (eth.iloc[idx+45]["Close"] - close_price) / close_price,
            "60_day_return": (eth.iloc[idx+60]["Close"] - close_price) / close_price,
            "75_day_return": (eth.iloc[idx+75]["Close"] - close_price) / close_price,
            "90_day_return": (eth.iloc[idx+90]["Close"] - close_price) / close_price,
            "120_day_return": (eth.iloc[idx+120]["Close"] - close_price) / close_price,
            "150_day_return": (eth.iloc[idx+150]["Close"] - close_price) / close_price,
            "180_day_return": (eth.iloc[idx+180]["Close"] - close_price) / close_price,
            "365_day_return": (eth.iloc[idx+365]["Close"] - close_price) / close_price
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
fig, ax = plt.subplots(figsize=(30,12))
ax.plot(eth["Close"], label="Close", linewidth = 3.5)
ax.plot(eth["365_day_MA"], label="365 day moving average", linewidth = 3.5)

# Add vertical lines for dates
for date in dates:
    ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

# Add legend and title
ax.legend(fontsize=22)
ax.set_title("Ethereum Price and 365D Moving Averages", fontsize=28, fontweight = 'bold')
ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=22, width=2, length=3)
ax.set_xlabel('Date', fontsize=30, fontweight = 'bold')
ax.set_ylabel('Price', fontsize=30, fontweight = 'bold')

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_plot.png"
plt.savefig(file_path)

# Plotting
fig, ax = plt.subplots(figsize=(22, 11))
bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3)

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
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_average_returns.png"
plt.savefig(file_path)

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
ax3.set_title("Number of positive and negative returns after ETH price cross below 365 days moving averages line", fontsize=20, fontweight='bold')
ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
ax3.set_xlabel("period", fontsize=20, weight='bold')
ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
# Rotate x-axis labels to normal
plt.xticks(rotation=45)

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_number_pos_neg_returns.png"
plt.savefig(file_path)

if eth.iloc[len(eth)-2]["Close"] > eth.iloc[len(eth)-2]["365_day_MA"]:
   if eth.iloc[len(eth)-1]["Close"] < eth.iloc[len(eth)-1]["365_day_MA"]:
      token = 'X'
      msg_type = 'sendMessage'
      chat_id = '-4240308218'
      text = 'Attention, ETH price go below 365D MA!'
      msg = f'https://api.telegram.org/bot{token}/{msg_type}?chat_id={chat_id}&text={text}'
      telegram_msg = requests.get(msg)

      # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a timeseries chart ETH price and 365D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_plot.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))

        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This are a average returns after ETH goes below 365D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_average_returns.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))


        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a number of positive and negative returns after ETH goes below 365D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_number_pos_neg_returns.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))
      
########### ETH Price above 365DMA###############################

# Get Bitcoin's data from yfinance library
eth = yf.Ticker("ETH-USD").history(period="max")

# Calculate 50, 100, and 200 day moving average
eth["50_day_MA"] = eth["Close"].rolling(window=50).mean()
eth["100_day_MA"] = eth["Close"].rolling(window=100).mean()
eth["365_day_MA"] = eth["Close"].rolling(window=365).mean()

dates = []
for i in range(len(eth) - 15):
    if eth.iloc[i]["Close"] < eth.iloc[i]["365_day_MA"]:
        if eth.iloc[i+1]["Close"] > eth.iloc[i+1]["365_day_MA"]:
            if (eth.iloc[i+1:i+16]["Close"] > eth.iloc[i+1:i+16]["365_day_MA"]).all():
                dates.append(eth.index[i+1])

returns = []
for date in dates:
    idx = eth.index.get_loc(date)
    close_price = eth.iloc[idx]["Close"]

    if idx + 365 < len(eth):
        returns.append({
            "date": date,
            "30_day_return": (eth.iloc[idx+30]["Close"] - close_price) / close_price,
            "45_day_return": (eth.iloc[idx+45]["Close"] - close_price) / close_price,
            "60_day_return": (eth.iloc[idx+60]["Close"] - close_price) / close_price,
            "75_day_return": (eth.iloc[idx+75]["Close"] - close_price) / close_price,
            "90_day_return": (eth.iloc[idx+90]["Close"] - close_price) / close_price,
            "120_day_return": (eth.iloc[idx+120]["Close"] - close_price) / close_price,
            "150_day_return": (eth.iloc[idx+150]["Close"] - close_price) / close_price,
            "180_day_return": (eth.iloc[idx+180]["Close"] - close_price) / close_price,
            "365_day_return": (eth.iloc[idx+365]["Close"] - close_price) / close_price
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
fig, ax = plt.subplots(figsize=(30,12))
ax.plot(eth["Close"], label="Close", linewidth = 3.5)
ax.plot(eth["365_day_MA"], label="365 day moving average", linewidth = 3.5)

# Add vertical lines for dates
for date in dates:
    ax.axvline(x=date, color='black', linestyle='--', alpha=0.9)

# Add legend and title
ax.legend(fontsize=22)
ax.set_title("Ethereum Price and 365D Moving Averages", fontsize=28, fontweight = 'bold')
ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=22, width=2, length=3)
ax.set_xlabel('Date', fontsize=30, fontweight = 'bold')
ax.set_ylabel('Price', fontsize=30, fontweight = 'bold')

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_plot2.png"
plt.savefig(file_path)

# Plotting
fig, ax = plt.subplots(figsize=(22, 11))
bars = average_returns.plot(kind='bar', ax=ax, edgecolor='black', linewidth=3)

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
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_average_returns2.png"
plt.savefig(file_path)

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
ax3.set_title("Number of positive and negative returns after ETH price cross above 365 days moving averages line", fontsize=20, fontweight='bold')
ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
ax3.tick_params(axis='both', which='minor', labelsize=20, width=2, length=3)
ax3.set_xlabel("period", fontsize=20, weight='bold')
ax3.set_ylabel("number of returns", fontsize=20, weight='bold')
plt.xticks(np.arange(9), ['30 days', '45 days', '60 days', '75 days', '90 days', '120 days', '150 days', '180 days','365 days'])
# Rotate x-axis labels to normal
plt.xticks(rotation=45)

# Save plot to Google Drive
file_path = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_number_pos_neg_returns2.png"
plt.savefig(file_path)

if eth.iloc[len(eth)-2]["Close"] < eth.iloc[len(eth)-2]["365_day_MA"]:
   if eth.iloc[len(eth)-1]["Close"] > eth.iloc[len(eth)-1]["365_day_MA"]:
      token = 'X'
      msg_type = 'sendMessage'
      chat_id = '-4240308218'
      text = 'Attention, ETH price go above 365D MA!'
      msg = f'https://api.telegram.org/bot{token}/{msg_type}?chat_id={chat_id}&text={text}'
      telegram_msg = requests.get(msg)

      # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This is a timeseries chart ETH price and 365D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_plot2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))

        # Send photo by URL with caption - Option 1
      token = 'X'
      msg_type = 'sendPhoto'
      chat_id = '-4240308218'
      caption = 'This are a average returns after ETH goes above 365D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_average_returns2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))


        # Send photo by URL with caption - Option 1
      token = 'X'
      chat_id = '-4240308218'
      caption = 'This is a number of positive and negative returns after ETH goes above 365D MA 😉'
      local_img = rf"C:\Users\Acer\OneDrive\Desktop\BTC_ETH_crossings\eth_close_365dma_number_pos_neg_returns2.png" # your image path

      telegram_msg = requests.post('https://api.telegram.org/bot'+ token + '/' + 'sendPhoto',
                             data={'chat_id': chat_id, 'caption' : caption},
                             files = ({'photo': open(local_img, 'rb')}))