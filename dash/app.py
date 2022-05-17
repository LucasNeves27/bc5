import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State


import numpy as np
import pandas as pd
from datetime import timedelta, datetime

import plotly.graph_objs as go

import json


from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

PROD = False

external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Open+Sans&display=swap',
    'https://fonts.googleapis.com/css2?family=Roboto:wght@400;900&display=swap',
    'https://fonts.googleapis.com/css2?family=Overlock:wght@900&display=swap',
    'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@600&display=swap'
]

COLORS = {
    "bg-secondary-color": "#1e1e2f",
    "bg-color": "#27293d",
    "color-primary": "#00BBE0",
    "color-lightGrey": "#d2d6dd",
    "color-grey": "#747681",
    "color-darkGrey": "#353a53",
    "font-color": "#c2c2d2"

}

LOADING_DIV = html.Div("Loading", className="loading-container")

###############################################################
## Functions
###############################################################


def get_info_value(info, keyname):
    infoValue = "" 
    if keyname in info.keys():
        infoValue = info[keyname] 
    
    return infoValue

def get_findata(sym, start_date, end_date):
    if PROD == True:
        df_ = pdr.get_data_yahoo(sym, start=start_date, end=end_date)
        df_ = df_.reset_index().set_index('Date').asfreq('d')
        df_['DateCol'] = df_.index
        df_.fillna(method='ffill', inplace=True)

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


def shift_split_data(df_, target_col, fitsize=7):
    df = df_.copy()
    ## Date_Y is the date being predicted
    ## The corresponding Date_X of the same row is the previous date
    df['Date_Y'] = df['DateCol']+ pd.DateOffset(days=1) 
    df['Y'] = df[target_col].shift(periods=-1)
    df.rename(columns={'DateCol':'Date_X'}, inplace=True)
    
    ## last `fitsize` days to fit for prediction on the next day
    df_fit = df.iloc[-fitsize:]
    
    return df_fit
    
def tidy_plot(fig_):
    fig_.update_traces(showlegend=False)
    fig_.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    fig_.update_yaxes(gridcolor=COLORS['color-darkGrey'])
    fig_.update_xaxes(gridcolor=COLORS['color-darkGrey'])
    
    return fig_

###############################################################
## Wrangle the data
###############################################################

date_interval = 100
END_DATE = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
START_DATE = (datetime.today() + timedelta(days=-date_interval)).strftime('%Y-%m-%d')



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
       options=['BTC-USD','ETH-USD', 'WIX', 'GOOG'],
       multi=False,
       value='WIX'
   )





###############################################################
## APP
###############################################################
title = "[APEX/DASH]"

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
                            html.H2(id='timeseries_title'),
                            html.P(id='timeseries_longname'),
                        ], className="timeseries_text"),
                    ]),
                    html.Div([
                        dcc.Loading(
                            id="loading-timeseries",
                            type="default",
                            children=[dcc.Graph(id='timeseries-dcc', style={'margin': '0'})]
                        ),
                        
                        ], className="timeseries_plot"
                    ),

                ], className='col timeseries_container'),

            ], className='row row-1'),

        ], className='card'),

        ########## Second Row ##########
        html.Div([

        html.Div([

                html.Div([
                    html.H3(id='techanalysis_title'),
                    html.P("Lorem Ipsum is awesome stuff"),
                    html.Div(
                        [dcc.Graph(id='techanalysis_dcc', style={'margin': '0'})],
                        className="techanalysis_container"
                    ),

                ], className='col'),

            ], className='row row-2'),

        ], className='card'),



        ########## Third Row ##########
        html.Div([

            html.Div([
                html.Div([
                    html.Div([
                            html.Div([
                                html.H3("Header Here")
                                ], className="country_profile_label"),
                            html.Div([
                                
                            ], id='country_selectorx', className="country_profile_selector"),
                        ], className="country_profile_title"),

                ], className='col-7'),
                html.Div(className='col'),
            ], className="row"),

            html.Div([
                html.Div([
                    
                    html.Div([
                        html.H4("Awesome Title"),
                        html.P(html.Em("Culpa aliqua culpa velit laboris sit sunt est laboris duis anim culpa.")),

                        html.Div([
                            dcc.Graph(id='sunburst_sources', style={'width': '95%', 'margin': '0 auto'}),

                        ], className='sunburst_container'
                        ),
                    ]),
                    ],
                    className='col-6'),
                html.Div([
                    html.Div([
                            html.H4(["Lorem Ipsum", html.Br(), html.Span(id='country_selection')]),
                            html.P("Anim elit proident proident exercitation cillum cillum nisi sit aliquip commodo."),
                            dcc.Graph(id='gdp_pct_ts', style={'margin': '0'}),
                            

                            html.H4(["Foo Bar Baz ", html.Span(id='country_selection2')]),
                            html.P("Labore aute duis voluptate veniam voluptate anim mollit cupidatat ipsum ipsum exercitation irure."),
                            dcc.Graph(id='pct_ts', style={'margin': '0'}),
                            

                    ], ),
                    ],
                    className='col ranking_container'),


            ],
            className='row'),

        ], className='card'),

        ################### References ###################
        html.Div([
            html.Div([
                html.Div([
                    html.H4("Apex Pattern Deployers"),
                    html.P("Kinney / Mendes / Neves / Pontejos")
                ], className='col-6 authors'),

                html.Div([
                    html.H5("Data Sources"),
                    html.H5("Assets Used"),
                ], className='col sources')
            ], className='row'),    
        ], className='card'),








    ], className="col-9"),

    ########## End Main Body ##########    

    ########## Sidebar ##########

        html.Div([
            
            html.H3("Symbol Selector", className="caption"),
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
            html.Div([
                # smiley_neutral
                html.Img(src="./assets/icons/frown.svg"),
                html.Img(src="./assets/icons/meh.svg"),
                html.Img(src="./assets/icons/smile.svg"),

            ], className="sentiment-container card"),

        ], className="col-3 sidebar"),

    ########## End Sidebar ##########


    ], className="row"),




        ],
        className="container"    
    ),








], className='outer')


######################################################Callbacks#########################################################


@app.callback(
    Output('timeseries_title', 'children'),
    Output('timeseries-dcc','figure'),
    #Output('loading-timeseries','children'),
    Output('timeseries_longname', 'children'),
    Output('company_profile', 'children'),

    Input(dropdown_symbols, 'value')
)
def getTimeSeriesPlot(ticker_symbol):
    fin_data, fin_info = get_findata(ticker_symbol, START_DATE, END_DATE)
    ## Split X and Y
    #split_fin_data = shift_split_data(fin_data, 'Close')
    #Y = split_fin_data['Close']
    #x_cols = [i for i in split_fin_data.columns.tolist() if i not in ['Date_X', 'Date_Y', 'Y'] ]
    #X = split_fin_data[x_cols]

    longName = get_info_value(fin_info, 'longName')
    longName = longName if len(longName) > 1 else ticker_symbol
    description = get_info_value(fin_info, 'description')

    industry = get_info_value(fin_info, 'industry')
    sector = get_info_value(fin_info, 'sector')
    logoImg = html.Span() 
    if len(get_info_value(fin_info, 'logo_url')) > 1 :
        logoImg = html.Img(src=get_info_value(fin_info, 'logo_url'))

    
    fig_ts = go.Figure(layout=default_layout)
    fig_ts.add_trace(go.Scatter(x=fin_data.index, y=fin_data['Close'],
                        mode='lines',
                        marker=dict(color=COLORS['color-primary'])
                        ))

    fig_ts = tidy_plot(fig_ts)
    fig_ts.update_xaxes(showspikes=True)
    fig_ts.update_xaxes(spikethickness=1)
    fig_ts.update_yaxes(showspikes=True)
    fig_ts.update_yaxes(spikethickness=1)
    
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
            html.P(description, style={"text-align":"left", "fontSize":"smaller"})
        ]
    
    
    return [ticker_symbol, fig_ts, longName, profile_details]

if __name__ == '__main__':
    app.run_server(debug=True)