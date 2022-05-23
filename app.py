import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State


import numpy as np
import pandas as pd
from datetime import timedelta, datetime

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import json
import re
import os


from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import ta 

import tweepy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import warnings
warnings.filterwarnings('ignore')


from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

###############################################################
## Some constant vars
###############################################################

date_interval = 365
END_DATE = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
START_DATE = (datetime.today() + timedelta(days=-date_interval)).strftime('%Y-%m-%d')

symbol_opts = pd.read_csv('./data/assets_list.csv')['Symbol'].tolist() + \
              pd.read_csv('./data/co_assets_list.csv')['Symbol'].tolist()


PROD = True
#PROD = False

ENABLETWEETS = True
TWEETSTOGET = 500

try:
    nltk.data.find('vader_lexicon')
    
except:
    nltk.download('vader_lexicon')


external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Roboto:wght@400;900&display=swap',
    'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@600&display=swap'
]

COLORS = {
    "bg-secondary-color": "#1e1e2f",
    "bg-color": "#27293d",
    "color-primary": "#00BBE0",
    "color-lightGrey": "#d2d6dd",
    "color-grey": "#747681",
    "color-darkGrey": "#353a53",
    "font-color": "#c2c2d2",
    "color-yellow":  "#FFCD3F",
    'color-red': "#FF3041",
    "color-purple":'#731DD8',
    "color-green":'#3bb001',
    "color-accent": "#f10075",
    "color-lime": "#affc41",
    "color-orange": "#fb8b24"

}

LOADING_DIV = html.Div("Loading", className="loading-container")

TAFEATS = [
    "Open", "High", "Low", "Close", "Volume", 
    'momentum_rsi', 
    'trend_macd_signal', 'trend_macd_diff', 'trend_macd', 
    'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 
    'volatility_atr', 'volume_obv', 'DateCol'
]

RANDOM_STATE = 9
score_func = 'neg_mean_absolute_percentage_error'

###############################################################
## Functions
###############################################################

def get_app_version():
    try:
        v = os.environ['GAE_VERSION']
        return html.P(v)
    except:
        return html.P("development")
    
###############################################################
## Utility Functions for dealing with data
###############################################################
def get_info_value(info, keyname):
    """
    Check whether key exists in Ticker.info and only access if it does; 
    otherwise return empty string.
    """
    infoValue = "" 
    if keyname in info.keys():
        infoValue = info[keyname] 
    
    return infoValue

def prep_findata(df_):
    """
    Fill nan rows: 
        Volume = 0
        Close = ffill
        Open, High, Low = Close
    """
    df_ = df_.reset_index().set_index('Date').asfreq('d')
    df_['DateCol'] = df_.index

    df_[['Close']] = df_[['Close']].fillna(method='ffill')
    df_.loc[df_['Open'].isna(), ['Volume']] = 0

    df_['tmp'] = df_['High']

    df_.loc[df_['High'].isna(), ['High']] = df_.loc[df_['tmp'].isna(), 'Close']
    df_.loc[df_['Open'].isna(), ['Open']] = df_.loc[df_['tmp'].isna(), 'Close']
    df_.loc[df_['Low'].isna(), ['Low']] = df_.loc[df_['tmp'].isna(), 'Close']

    df_.drop(columns=["tmp","Adj Close"], inplace=True)

    return df_

def get_findata(sym, start_date, end_date):
    """
    Returns Ticker.info if PROD==True, 
    otherwise return local sample info.json file
    """
    if PROD == True:
        df_ = pdr.get_data_yahoo(sym, start=start_date, end=end_date)
        df_ = prep_findata(df_)
        json_object = yf.Ticker(sym).info
        return df_, json_object
    else:
        df_ = pd.read_csv('./data/fin_data.csv')
        df_['Date'] = pd.to_datetime(df_['Date'])
        df_['DateCol'] = pd.to_datetime(df_['DateCol'])

        df_ = df_.set_index('Date').asfreq('d')
        json_object = {}

        with open('./data/fin_info.json', 'r') as fp:
            json_object = json.load(fp)
        return df_, json_object




# from https://levelup.gitconnected.com/how-to-format-integers-into-string-representations-in-python-9f6ad0f2d36f
def safe_num(num):
    if isinstance(num, str):
        num = float(num)
    return float('{:.3g}'.format(abs(num)))

# also from https://levelup.gitconnected.com/how-to-format-integers-into-string-representations-in-python-9f6ad0f2d36f
def format_number(num):
    num = safe_num(num)
    sign = ''

    metric = {'T': 1000000000000, 'B': 1000000000, 'M': 1000000, 'K': 1000, '': 1}

    for index in metric:
        num_check = num / metric[index]

        if(num_check >= 1):
            num = num_check
            sign = index
            break
    if num == 0:
        return ""
    else:
        return f"{str(num).rstrip('0').rstrip('.')} {sign}"


###############################################################
## Apply model
###############################################################


def shift_split_data(df_, target_col, fitsize=30):
    """
    Shifts the "Y" column so that the X columns correspond
    to the next day Y as prediction.
    """
    df = df_.copy()
    ## Date_Y is the date being predicted
    ## The corresponding Date_X of the same row is the previous date
    df['Date_Y'] = df['DateCol']+ pd.DateOffset(days=1) 
    df['Y'] = df[target_col].shift(periods=-1)
    df.rename(columns={'DateCol':'Date_X'}, inplace=True)
    
    ## last fitsize days to fit for prediction on the next day
    df_fit = df.iloc[-fitsize-1:]
    
    return df_fit

def make_prediction(df_fit):
    """
    Apply ML model to predict. 
    Use GridSearch so we can retrieve a MAPE score if necessary,
    but use previously obtained hyperparameters
    """
    ## Split X and Y
    Y = df_fit['Y'][:-1]
    x_cols = [i for i in df_fit.columns.tolist() if i not in ['Date_X', 'Date_Y', 'Y'] ]
    X = df_fit[x_cols][:-1]
    X_predict = df_fit[x_cols][-1:]

    params_gbr = {
        'gradientboostingregressor__learning_rate': [.05],
        'gradientboostingregressor__loss': ['quantile'],
        'gradientboostingregressor__max_depth': [1],
        'gradientboostingregressor__max_features': ['auto'],
        'gradientboostingregressor__n_estimators': [50],
        'selectkbest__k': [14]
    }

    pipe_gbr = make_pipeline(StandardScaler(),
                        SelectKBest(f_regression),
                        GradientBoostingRegressor(random_state=RANDOM_STATE))

    tscv = TimeSeriesSplit(n_splits=10, max_train_size=7, test_size=1)
    # Setting up the grid search
    gs_gbr = GridSearchCV(pipe_gbr, 
                        params_gbr, 
                        n_jobs=-1, 
                        cv=tscv,
                        scoring=score_func,
                        refit=True
                        )

    gs_gbr.fit(X, Y)
    pred = gs_gbr.best_estimator_.predict(X_predict)[0]

    return pred

###############################################################
## Utility Functions for dealing with plots
###############################################################
def tidy_plot(fig_):
    """
    Apply some default aesthetics to plots
    """
    fig_.update_traces(showlegend=False)
    fig_.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    fig_.update_yaxes(gridcolor=COLORS['color-darkGrey'])
    fig_.update_xaxes(gridcolor=COLORS['color-darkGrey'])
    fig_.update_xaxes(showspikes=True, spikethickness=1)
    fig_.update_yaxes(showspikes=True, spikethickness=1)
    fig_.update_yaxes(zerolinewidth=1, zerolinecolor='rgba(255,255,255,.5)')
    return fig_


def make_sparklines():
    """
    Render the 10 sparklines 
    """
    sp10_prices = get_sp10()

    fig_sub = make_subplots(rows=10, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in range(len(sp10_prices.columns.tolist())):
        sym_i = sp10_prices.columns.tolist()[i]
        
        fig_sub.append_trace(go.Scatter(
            y=sp10_prices.iloc[:,i],
            x=sp10_prices.index,
            name=sym_i
        ), row=i+1, col=1)
        fig_sub.update_yaxes(row=i+1, col=1, showticklabels=False)
        
    fig_sub.update_layout(
        paper_bgcolor=COLORS['bg-color'],
        plot_bgcolor=COLORS['bg-color'],
        font_color=COLORS['font-color'],
        )
    fig_sub.update_layout(height=500, width=300, showlegend=True,)
    fig_sub.update_layout(margin = dict(t=30, l=0, r=0, b=0),) 
    fig_sub.update_layout(title_text="30-Day Performance of Top 10 Assets",
                        font=dict(
                            size=10,
                        ))
    fig_sub.update_yaxes(showgrid=False)
    fig_sub.update_xaxes(showgrid=False)

    return fig_sub


def make_treemap(fig_tm):
    sp500 = pd.read_csv('./data/marketcap.csv')
    sp500_sectors = sp500.groupby(['Sector']).sum().reset_index()
    sp500_sectors[['Text']] = sp500_sectors[['MarketCap']]
    sp500_sectors[['MarketCap']] = 0
    sp500_sectors['Symbol'] = sp500_sectors['Sector']
    sp500_sectors['Name'] = sp500_sectors['Sector']
    sp500_sectors.drop(columns=['Sector'], inplace=True)
    sp500_sectors['Sector'] = "SP500"
    sp500['Text'] = sp500['MarketCap']

    sp500_tree = pd.concat([sp500_sectors, sp500.loc[sp500['MarketCap']>0,['Sector','MarketCap','Symbol','Name', 'Text']]]).reset_index(drop=True)
    sp500_tree.rename(columns={'Sector':'Parent', 'Symbol':"Label"}, inplace=True)
    sp500_tree['Text'] = sp500_tree['Text'].apply(format_number)
    sp500_tree_ = pd.concat([pd.DataFrame([[0, "", "SP500","SP500",""]], columns=sp500_tree.columns),sp500_tree])

    customdata = np.dstack((sp500_tree_[['Text']],sp500_tree_[['Name']]))

    fig_tm.add_trace(go.Treemap(
        labels = sp500_tree_['Label'],
        parents = sp500_tree_['Parent'],
        values=sp500_tree_['MarketCap'],
        customdata=customdata,
        hovertemplate="<b>%{customdata[0][1]}</b><br>%{customdata[0][0]} <extra></extra>",
    ))
    fig_tm.update_layout(
        height=500,
        width=500,
        uniformtext=dict(minsize=12, mode='hide'),
        treemapcolorway=[
            COLORS["color-yellow"],
            COLORS["color-accent"],
            COLORS["color-orange"],
            COLORS["color-lime"],
            COLORS["color-primary"],
            COLORS["color-green"],
        ]
    )
    fig_tm.update_layout(margin = dict(t=10, l=0, r=0, b=0))




    return fig_tm

def make_bb_plots(fin_data_, fig_):

    bb_cols = ['Close', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl', ]
    bb_colors = [COLORS['color-primary'], COLORS['color-red'], COLORS['color-yellow'], COLORS['color-yellow'], ]
    plot_opts = [{'line_width': 1},
                 {'line_width': 1},
                 {'line_width': 1, 'line_dash':"dot"},
                 {'line_width': 1,'line_dash':"dot"},
                 ]
    make_ta_plots(fin_data_, fig_, bb_cols, bb_colors, plot_opts)
    
    fig_ = tidy_plot(fig_)
    fig_.update_traces(showlegend=True)
    fig_.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ))

    return fig_

def make_macd_plots(fin_data_, fig_):
    macd_cols = ['trend_macd_diff', 'trend_macd', 'trend_macd_signal']
    macd_colors = [ COLORS['color-lime'], COLORS['color-orange'], COLORS['color-yellow']]
    plot_opts = [{'line_width': 1, 'line_dash':"dot"}, 
                 {'line_width': 1, 'line_dash':"dot"}, 
                 {'line_width': 1, }]
    make_ta_plots(fin_data_, fig_, macd_cols, macd_colors, plot_opts)

    fig_.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='rgba(255,255,255,.5)')
    fig_ = tidy_plot(fig_)
    fig_.update_traces(showlegend=True)
    fig_.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ))
    return fig_

def make_rsi_plots(fin_data_, fig_):
    macd_cols = ['momentum_rsi',]
    macd_colors = [ COLORS['color-orange'], ]
    plot_opts = [{'line_width': 1}]
    make_ta_plots(fin_data_, fig_, macd_cols, macd_colors, plot_opts)
    fig_ = tidy_plot(fig_)

    return fig_


def make_obv_plots(fin_data_, fig_):
    macd_cols = ['volume_obv',]
    macd_colors = [ COLORS['color-orange'], ]
    plot_opts = [{'line_width': 1}]
    make_ta_plots(fin_data_, fig_, macd_cols, macd_colors, plot_opts)
    fig_ = tidy_plot(fig_)

    return fig_

def make_vol_plots(fin_data_, fig_):
    macd_cols = ['volatility_atr',]
    macd_colors = [ COLORS['color-orange'], ]
    plot_opts = [{'line_width': 1}]
    make_ta_plots(fin_data_, fig_, macd_cols, macd_colors, plot_opts)
    fig_ = tidy_plot(fig_)

    return fig_

def make_ta_plots(data_, fig_, cols_, colors_, plot_opts):

    for i in range(len(cols_)):
        fig_.add_trace(go.Scatter(x=data_.index, y=data_[cols_[i]],
                    mode='lines',
                    marker=dict(color=colors_[i]),
                    name=str.upper(str.replace(cols_[i], "_", " ")),
                    **plot_opts[i]
                    ))
    return fig_

def get_sp10():
    """
    Fetch the 30-day data for the top 10 assets by marketcap if PROD==True
    otherwise retrieve local sample data.
    """
    sp10 = pd.read_csv('./data/marketcap.csv')
    sp10 = sp10.sort_values(by='MarketCap', ascending=False).reset_index(drop=True)

    sp10_list = sp10[0:10]['Symbol'].tolist()
    today = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    prevmonth = (datetime.today() + timedelta(days=-30)).strftime('%Y-%m-%d')

    if PROD == True:
        sp10_prices = pdr.get_data_yahoo(sp10_list[0], start=prevmonth, end=today)[['Close']].rename(columns={'Close':sp10_list[0]})

        for i in sp10_list[1:]:
            sp10_prices[i] = pdr.get_data_yahoo(i, start=prevmonth, end=today)[['Close']].rename(columns={'Close':i})
        return sp10_prices
    
    else: 
        sp10_prices = pd.read_csv('./data/sp10prices.csv').set_index('Date')
        return sp10_prices


###########################
## Sentiment Analysis
###########################

def get_tweets(sym):
    """
    Fetch tweets from Twitter API if PROD==True, and ENABLETWEETS==True
    otherwise retrieve from local sample twitter file
    """
    apiKey = ""
    apiSecret = ""
    accessToken = ""
    accessTokenSecret = ""
    
    if (PROD == True) and (ENABLETWEETS == True):
        tk = pd.read_csv('./twitter.csv')
        apiKey = tk.loc[tk['label']=='api']['key'].values[0]
        apiSecret = tk.loc[tk['label']=='apisecret']['key'].values[0]
        accessToken = tk.loc[tk['label']=='accesstoken']['key'].values[0]
        accessTokenSecret = tk.loc[tk['label']=='accesstokensecret']['key'].values[0]
        
        tweet_list = []

        try:
            auth = tweepy.OAuthHandler(apiKey, apiSecret)
            auth.set_access_token(accessToken, accessTokenSecret)
            api = tweepy.API(auth)

            tweets = tweepy.Cursor(api.search_tweets, q=sym, tweet_mode='extended', lang='en').items(TWEETSTOGET)
            for t in tweets:
                if not "bot" in str.lower(t.author.name) :
                    tweet_list.append(t.full_text)

            return pd.DataFrame(tweet_list)
        except:
            pass
    else:
        return pd.read_csv('./data/tweet_sample.csv')



def get_sentiment(s):
    """
    Return Sentiment scores
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(s)
    neg = scores['neg']
    neu = scores['neu']
    pos = scores['pos']
    
    return [neg, neu, pos]

def clean_text(txt):
    """
    Perform basic text processing
    """
    txt = re.sub(r"RT[\s]+", "", txt)
    txt = txt.replace("\n", " ")
    txt = re.sub(" +", " ", txt)
    txt = re.sub(r"https?:\/\/\S+", "", txt)
    txt = re.sub(r"(@[A-Za-z0–9_]+)|[^\w\s]|#", "", txt)
    #txt = emoji.replace_emoji(txt, replace='')
    txt.strip()
    return txt

def get_tweets_sentiment(sym):
    """
    Calculates the sentiment of each tweet retrieved. 
    Returns scores.
    """

    tweets = get_tweets(sym)

    tweets.columns = ["tweets"]
    tweets["tweets"] = tweets["tweets"].apply(clean_text)
    
    res = pd.DataFrame(tweets['tweets'].apply(lambda x: get_sentiment(x))) 
    res = res.apply(lambda row: row['tweets'], axis=1, result_type='expand').rename(columns={0:'neg', 1:'neu', 2:'pos'})
    scores = {'neg': res.mean()['neg'], 'neu': res.mean()['neu'], 'pos': res.mean()['pos'] }
    

    return scores



###############################################################
## Layouts
###############################################################

default_layout = go.Layout(
        height=300,
        paper_bgcolor=COLORS['bg-color'],
        plot_bgcolor=COLORS['bg-color'],
        font_color=COLORS['font-color'],
        )

###############################################################
## Interactive Components
###############################################################

dropdown_symbols = dcc.Dropdown(
       id='cc_drop',
       options=symbol_opts,
       multi=False,
       value='GOOG'
   )



###############################################################
## APP
###############################################################
title = "[APEX·DASH]"

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.css.config.serve_locally = True
app.title = title
app._favicon = ('logo.png')

server = app.server

app.layout = html.Div([

    ########## Nav ##########
    html.Div([
        html.Nav([
            html.Div([
                html.A([
                    html.Img(src='/assets/logo.png'), 
                    html.H1(title)
                    ], className="brand",),

            ], className="nav-left"),
            html.Div([], className='nav-center'),
            html.Div([
                html.Div([
                    html.P(datetime.today().strftime('%d %b %Y'),id="rightnow_day"),
                ], className="rightnow"),
            ], className='nav-right'),
        ], className='nav main-nav container'),
    ], className="nav-container"),

    ########## End Nav ##########

    html.Div([
    html.Div([

    ########## Main Body ##########    
    html.Div([
        
        ########## First Row ##########

        html.Div([
            html.Div([
                html.Div([
                    dcc.Loading(
                        id="loading-timeseries-text",
                        children=[
                            html.Div([
                                
                                html.Div([
                                    html.H2(id='timeseries_title'),
                                    html.P(id='timeseries_longname'),
                                ], className="timeseries_text"),
                                html.Div([
                                    html.Div([
                                        html.Span("Last Close: ", id="lastclose_label"),
                                        html.Span([], id="lastclose_value"),
                                    ]),
                                    html.Div([
                                        html.Span("Prediction: ", id="prediction_label"),
                                        html.Span([], id="prediction_value"),
                                    ]),
                                ], className="prediction_text"),
                            
                            ], className="timeseries_text_container"),
                    ]),

                    dcc.Tabs([
                        dcc.Tab(label='Timeseries', children=[
                            dcc.Loading(
                                id="loading-timeseries",
                                type="default",
                                children=[dcc.Graph(id='timeseries-dcc')]
                            ),
                        ], className="ta-tab", selected_className="ta-tab-selected"),
                        dcc.Tab(label='Candlestick', children=[
                            dcc.Loading(
                                id="loading-candlestick",
                                type="default",
                                children=[dcc.Graph(id='candlestick-dcc')]
                            ),
                        ], className="ta-tab", selected_className="ta-tab-selected"),

                    ]),

                ], className='col timeseries_container'),

            ], className='row row-1'),

        ], className='card'),

        ########## Second Row ##########
        html.H3("Technical Analysis Indicators", className="caption"),
        html.Div([

        html.Div([ ## ROW open

                html.Div([ ## COL open
                    ######### Begin Tabbed Area #########
                    html.Div([

                        dcc.Tabs([
                            dcc.Tab(label='BB', children=[
                                html.H3("Bollinger Bands", id='techanalysis_title'),
                                dcc.Loading(
                                    id="loading-ta",
                                    type="default",
                                    children=[
                                        dcc.Graph(id='techanalysis-dcc', style={'margin': '0'})
                                    ]),

                            ], className="ta-tab", selected_className="ta-tab-selected"),
                            dcc.Tab(label='MACD', children=[
                                html.H3("Moving Average Convergence Divergence", ),
                                dcc.Loading(
                                    id="loading-ta2",
                                    type="default",
                                    children=[
                                        dcc.Graph(id='techanalysis-dcc2', style={'margin': '0'})
                                    ]),
                            ], className="ta-tab", selected_className="ta-tab-selected"),

                            dcc.Tab(label='RSI', children=[
                                html.H3("Momentum: RSI", ),
                                dcc.Loading(
                                    id="loading-ta3",
                                    type="default",
                                    children=[
                                        dcc.Graph(id='techanalysis-dcc3', style={'margin': '0'})
                                ]),
                            ], className="ta-tab", selected_className="ta-tab-selected"),

                            dcc.Tab(label='ATR', children=[
                                html.H3("Volatility: ATR", ),
                                dcc.Loading(
                                    id="loading-ta4",
                                    type="default",
                                    children=[
                                        dcc.Graph(id='techanalysis-dcc4', style={'margin': '0'})
                                ]),
                            ], className="ta-tab", selected_className="ta-tab-selected"),

                            dcc.Tab(label='OBV', children=[
                                html.H3("On Balance Volume", ),
                                dcc.Loading(
                                    id="loading-ta5",
                                    type="default",
                                    children=[
                                        dcc.Graph(id='techanalysis-dcc5', style={'margin': '0'})
                                    ]),
                            ], className="ta-tab", selected_className="ta-tab-selected"),
                        ]), ## close dcc.Tabs
                    ], className="tabbed"),
                    
                    
                    ######### End Tabbed Area #########
                    

                ], className='col'),

            ], className='row row-2'),

        ], className='card'),
        
        ########## Third Row ##########
        
        html.H3("Market Performance", className="caption"),
        
        html.Div([
            html.Div([
                html.Div([
                    dcc.Loading(children=[
                        html.P("Market Capitalization", style={"margin":0}),
                        dcc.Graph(id='treemap-dcc'),
                    ], type="default"),
                ], className="col"),
                html.Div([
                    dcc.Loading(children=[
                        dcc.Graph(figure=make_sparklines()),
                    ], type="default"),
                ], className="col"),
            ], className='row'),
        ], className="card"),


    ], className="col-9 main"),

    ########## End Main Body ##########    

    ########## Sidebar ##########

        html.Div([
            
            html.H3("Symbol Selector", className="caption", style={"marginTop":0}),
            html.Div([
                    dropdown_symbols
            ], id='symbol_selector', className="symbol_selector_container"),

            html.H3("Stock Profile", className="caption"),
            dcc.Loading(
                id="loading-profile",
                type="default",
                children=html.Div(id='company_profile', className="card company-profile-container"),
            ),
            


            html.H3("Twitter Sentiment", className="caption"),
                dcc.Loading(children=[
                    html.Div(id="twitter_sentiment", className="sentiment-container card"),
                ], type="default"),


        ################### References ###################
        html.H3("External Links", className="caption"),
        html.Div([
            html.P("Data Sources", className="linkstitle"),
            html.Ul([
                html.Li(html.A("Daily stock data", href="https://finance.yahoo.com/")),
                html.Li(html.A("S&P constituents", href="https://github.com/datasets/s-and-p-500-companies")),
                html.Li(html.A("Tweets", href="https://twitter.com/")),
            ], className="sources"),
            html.P("Code", className="linkstitle"),
            html.Ul([
                html.Li(html.A("GitHub Repo", href="https://github.com/fpontejos/bc5")),
            ], className="sources"),
        ], className="card"),

        html.H3("The Team", className="caption"),
        
        html.Div([
            html.Div([
                html.Div([
                    html.H4("Apex Pattern Deployers"),
                ], className='apex-container'),
                html.Div([
                    html.Img(src="/assets/logo-branco.png"),
                    html.Div([
                        html.Span("Kinney · Mendes"), 
                        html.Span("Neves · Pontejos"),
                    ], className="author-names"),
                ], className="authors",), 
                html.Div(html.Em("We are not financial advisors and this is not financial advise."), className="footnote"),
                html.Div([
                    html.Img(src="/assets/logo.png"),
                    html.Div(get_app_version(), className="app-version"),
                ], className="app-details"),
            ], className='authors-container'),


        ], className='card'),


        ], className="col-3 sidebar"),

    ########## End Sidebar ##########


    ], className="row"),


    ], className="container"    
    ),



], className='outer')


######################################################Callbacks#########################################################


@app.callback(
    Output('timeseries_title', 'children'),
    Output('timeseries-dcc','figure'),
    Output('candlestick-dcc','figure'),
    Output('treemap-dcc','figure'),
    Output('techanalysis-dcc','figure'),
    Output('techanalysis-dcc2','figure'),
    Output('techanalysis-dcc3','figure'),
    Output('techanalysis-dcc4','figure'),
    Output('techanalysis-dcc5','figure'),
    Output('timeseries_longname', 'children'),
    Output('company_profile', 'children'),
    Output('twitter_sentiment', 'children'),
    Output('prediction_value', 'children'),
    Output('lastclose_value', 'children'),

    Input(dropdown_symbols, 'value')
)
def getTimeSeriesPlot(ticker_symbol):
    fin_data, fin_info = get_findata(ticker_symbol, START_DATE, END_DATE)
    fin_data['Volume'] = fin_data['Volume'].astype(float)

    fin_data_ta = ta.add_all_ta_features(
        fin_data, "Open", "High", "Low", "Close", "Volume", fillna=True
    )[TAFEATS]


    longName = get_info_value(fin_info, 'longName')
    longName = longName if len(longName) > 1 else ticker_symbol
    description = get_info_value(fin_info, 'description')

    industry = get_info_value(fin_info, 'industry')
    sector = get_info_value(fin_info, 'sector')
    logoImg = html.Span() 
    if len(get_info_value(fin_info, 'logo_url')) > 1 :
        logoImg = html.Img(src=get_info_value(fin_info, 'logo_url'))

    ##########################################################
    ## Make Timeseries Plot
    ##########################################################
    fig_ts = go.Figure(layout=default_layout)
    fig_ts.add_trace(go.Scatter(x=fin_data.index, y=fin_data['Close'],
                    mode='lines',
                    line_width=1,
                    marker=dict(color=COLORS['color-primary'])
                    ))

    fig_ts = tidy_plot(fig_ts)

    ##########################################################
    ## End Timeseries Plot
    ##########################################################
    
    ##########################################################
    ## Make Candlestick Plot
    ##########################################################

    fig_cs = go.Figure(go.Candlestick(
        x=fin_data.index,
        open=fin_data['Open'],
        high=fin_data['High'],
        low=fin_data['Low'],
        close=fin_data['Close'],

    ), layout=default_layout)

    fig_cs = tidy_plot(fig_cs)

    fig_cs.update_layout(xaxis_rangeslider_visible=False)
    fig_cs.update_traces(line={'width':1})

    ##########################################################
    ## End Candlestick Plot
    ##########################################################

    profile_details = [
            logoImg,
            html.Ul([
                html.Li([sector]),
                html.Li([industry]),
            ], className="profile-details-list")
    ]

    if len(sector) < 1 and len(industry) < 1 :
        profile_details = [
            html.P(html.Strong(ticker_symbol)),
            html.P(description, style={"textAlign":"left", "fontSize":"smaller"})
        ]
    


    ##########################################################
    ## Make TA Plots
    ##########################################################
    fig_dcc = go.Figure(layout=default_layout)
    fig_dcc = make_bb_plots(fin_data_ta, fig_dcc)

    fig_dcc2 = go.Figure(layout=default_layout)
    fig_dcc2 = make_macd_plots(fin_data_ta, fig_dcc2)

    fig_dcc3 = go.Figure(layout=default_layout)
    fig_dcc3 = make_rsi_plots(fin_data_ta, fig_dcc3)

    fig_dcc4 = go.Figure(layout=default_layout)
    fig_dcc4 = make_vol_plots(fin_data_ta, fig_dcc4)

    fig_dcc5 = go.Figure(layout=default_layout)
    fig_dcc5 = make_obv_plots(fin_data_ta, fig_dcc5)

    ##########################################################
    ## Make Treemap
    ##########################################################


    fig_tm = go.Figure(layout=default_layout)
    fig_tm = make_treemap(fig_tm)


    ##########################################################
    ## Get Tweets Sentiment
    ##########################################################
    twitter_sentiment_scores = get_tweets_sentiment(ticker_symbol)
    tweet_label = max(twitter_sentiment_scores, key=twitter_sentiment_scores.get)
    if tweet_label == 'neu':
        tweet_label = 'Neutral'
    elif tweet_label == 'neg':
        tweet_label = 'Negative'
    else:
        tweet_label = 'Positive'

    tw_neg = round(twitter_sentiment_scores['neg'],2)
    tw_neu = round(twitter_sentiment_scores['neu'],2)
    tw_pos = round(twitter_sentiment_scores['pos'],2)

    tweet_faces = [
        html.Div([
            html.Img(src="./assets/icons/frown.svg", style={'opacity': max(.2,tw_pos)}),
            html.Div(str(int(100*tw_neg))+"%", ),
        ], className="tweet_sentiment_result"),

        html.Div([
            html.Img(src="./assets/icons/meh.svg", style={'opacity': max(.2,tw_neu)}),
            html.Div(str(int(100*tw_neu))+"%", ),
        ], className="tweet_sentiment_result"),

        html.Div([
            html.Img(src="./assets/icons/smile.svg", style={'opacity': max(.2,tw_pos)}),
            html.Div(str(int(100*tw_pos))+"%", ),
        ], className="tweet_sentiment_result"),

        html.Div(tweet_label, className="tweet_sentiment_label")
    ]

    ##########################################################
    ## Make Predictions
    ##########################################################
    df_fit = shift_split_data(fin_data_ta,'Close')
    pred = make_prediction(df_fit)
    lastclose = fin_data_ta['Close'][-1:].values[0]
    
    pred_direction = "pred increase" if pred > lastclose else "pred decrease"
    pred = '{:.3f}'.format(round(pred,3))

    pred_element = html.Span(pred, className=pred_direction)
    lastclose_element = html.Span('{:.3f}'.format(round(lastclose,3)))
        
    
    return [ticker_symbol, fig_ts, fig_cs, fig_tm,
            fig_dcc, fig_dcc2, fig_dcc3, fig_dcc4, fig_dcc5, 
            longName, profile_details, tweet_faces, 
            pred_element, lastclose_element]



if __name__ == '__main__':
    app.run_server(debug=True)
