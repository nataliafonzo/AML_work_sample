import pandas as pd
import numpy as np
from datetime import datetime

from settings import keywords_list, stopwords_spanish, raw_columns
from nltk.tokenize import word_tokenize
import spacy
import re


def clean_nulls(dataset):
    """Replace empty lists or dicts with None in every dictionary
    contained in a list. Convert the list to dataframe. 
    Args:
        dataset (list): list of dicts
    Returns:
        dataframe (dataframe)
    """
    for i in range(len(dataset)):
        for key, value in dataset[i].items():
            if (value == [] or value == {}):
                dataset[i][key] = None
    dataframe = pd.DataFrame(dataset)
    return dataframe


# NLP auxiliar functions
def tokenizar(text):
    """Remove numbers and symbols from a text, convert letters to
    lowercase and tokenize it.
    Args: 
        text (str): text to be processed
    Returns:
        text (list): list of tokens
    """
    text = re.sub(r'[^\w]', ' ', text)
    text = re.sub('\d+', ' ', text)
    text = re.sub('  ',' ',text)
    text = word_tokenize(text.lower())
    return text

def drop_stopwords(tokens):
    """Remove stopwords from a list of tokens.
    Args: 
        tokens (list): list of tokens (str)
    Returns:
        tokens_clean (list): list of tokens (str)
    """
    is_non_stopword = [token not in stopwords_spanish for token in tokens]
    tokens_clean = np.array(tokens)[is_non_stopword]
    return tokens_clean

def build_NLP_features(dataframe):
    """Create features based on the text in "title" column.
    Args: 
        dataframe (dataframe): dataframe containing "title" column
    Returns:
        dataframe (dataframe): dataframe with added NLP features
    """
    dataframe['title_tok'] = [drop_stopwords(tokens) for tokens in [tokenizar(text) for text in dataframe.title]]
    # Creates binary features indicating whether certain keywords are found in the title or not
    for keyword_group in keywords_list:
        dataframe[keyword_group[0].upper()] = [int(any(keyword in list(title) for keyword in keyword_group)) for title in dataframe.title_tok]
    # Creates word2vec features
    nlp = spacy.load('es_core_news_sm')
    with nlp.disable_pipes():
        title_vectors = np.array([nlp(text).vector for text in dataframe.title])    
    title_vectors_T = title_vectors.transpose()
    for i in range(96):
        column_name = 'TITLE_VEC_' + str(i)
        dataframe = pd.concat((dataframe, pd.Series(title_vectors_T[i]).rename(column_name)), axis=1)
    return dataframe


def extract_features(dataframe):
    """Create features from dataframe.
    Args:
        dataframe (dataframe): dataframe version of X_train (or X_test)
        after applying clean_nulls
    Returns:
        datafra (dataframe)
    """
    #Auxiliar dates
    dataframe['last_update_date'] = [datetime.strptime(i[:10],'%Y-%m-%d') for i in dataframe.last_updated]
    dataframe['date_created_date'] = [datetime.strptime(i[:10],'%Y-%m-%d') for i in dataframe.date_created]
    dataframe['stop_time_date'] = [datetime.strptime(i[:10],'%Y-%m-%d') for i in dataframe.stop_time]
    dataframe['aux_creation_to_update'] = dataframe['last_update_date'] - dataframe['date_created_date']
    dataframe['aux_creation_to_stop'] = dataframe['stop_time_date'] - dataframe['date_created_date']
    #Numeric
    dataframe['CREATION_TO_UPDATE'] = [i.days for i in dataframe['aux_creation_to_update']]
    dataframe['CREATION_TO_STOP'] = [i.days for i in dataframe['aux_creation_to_stop']]
    dataframe['PRICE'] = dataframe.price
    dataframe['QTY_INITIAL'] = dataframe.initial_quantity
    dataframe['QTY_AVAILABLE'] = dataframe.available_quantity
    dataframe['QTY_SOLD'] = dataframe.sold_quantity
    dataframe['SHARE_SOLD'] = dataframe.QTY_SOLD / dataframe.QTY_INITIAL
    dataframe['QTY_VARIATIONS'] = [i if i == None else len(i) for i in dataframe.variations]
    dataframe['LEN_ATRIBUTOS'] = [None if i == None else len(i) for i in dataframe.attributes]
    dataframe['LATITUDE'] = [None if i['latitude'] == '' else i['latitude'] for i in dataframe.geolocation]
    dataframe['LONGITUDE'] = [None if i['longitude'] == '' else i['longitude'] for i in dataframe.geolocation]
    #Categorical
    dataframe['TAG_SUSPEND_STATUS'] = ['no_value' if i == None else i[0] for i in dataframe.sub_status]
    dataframe['TAG_BUY_MODE'] = ['no_value' if i == None else i for i in dataframe.buying_mode] 
    dataframe['TAG_LISTING'] = ['no_value' if i == None else i for i in dataframe.listing_type_id] 
    #Binary
    dataframe['FLAG_MERCADOPAGO'] = dataframe.accepts_mercadopago.astype(int)
    dataframe['FLAG_AUTO_RELIST'] = dataframe.automatic_relist.astype(int)
    dataframe['FLAG_BIDS_VISITS'] = [None if i == None else int('dragged_bids_and_visits' in i) for i in dataframe.tags]
    dataframe['FLAG_VISITS'] = [None if i == None else int('dragged_visits' in i) for i in dataframe['tags']]
    dataframe['FLAG_GOOD_THUMBNAIL'] = [None if i == None else int('good_quality_thumbnail' in i) for i in dataframe.tags]
    dataframe['FLAG_POOR_THUMBNAIL'] = [None if i == None else int('poor_quality_thumbnail' in i) for i in dataframe.tags]
    dataframe['FLAG_FREE_RELIST'] = [None if i == None else int('free_relist' in i) for i in dataframe.tags]
    dataframe['FLAG_VARIATIONS'] = [int(i != None) for i in dataframe.variations]
    dataframe['FLAG_GARANTIA'] = [int(i != None) for i in dataframe.warranty]
    dataframe['FLAG_CONTACT'] = [int(i != None) for i in dataframe.seller_contact]
    dataframe['FLAG_EMAIL'] = [int(not(i == None or i['email'] == '')) for i in dataframe.seller_contact]
    dataframe['FLAG_LOCAL_PICK_UP'] = [int(i['local_pick_up']) for i in dataframe.shipping]
    dataframe['FLAG_FREE_SHIPPING'] = [int(i['free_shipping']) for i in dataframe.shipping]
    dataframe['FLAG_LOCATION'] = [int(i != None) for i in dataframe.location]
    dataframe['FLAG_OPEN_HOURS'] = [i if i == None else int(i['open_hours'] != '') for i in dataframe.location]
    dataframe['ACCEPTS_EFECTIVO'] = [0 if i == None else int('Efectivo' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods] 
    dataframe['ACCEPTS_TRANSFERENCIA'] = [0 if i == None else int('Transferencia bancaria' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_TARJETA'] = [0 if i == None else int('Tarjeta de crédito' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_ACORDAR'] = [0 if i == None else int('Acordar con el comprador' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_GIRO'] = [0 if i == None else int('Giro postal' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_MP'] = [0 if i == None else int('MercadoPago' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_VISA'] = [0 if i == None else int('Visa' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_MASTER'] = [0 if i == None else int('MasterCard' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_REEMBOLSO'] = [0 if i == None else int('Contra reembolso' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_VISA_ELECTRON'] = [0 if i == None else int('Visa Electron' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_MAESTRO'] = [0 if i == None else int('Mastercard Maestro' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_AMERICAN'] = [0 if i == None else int('American Express' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_DINERS'] = [0 if i == None else int('Diners' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    dataframe['ACCEPTS_CHEQUE'] = [0 if i == None else int('Cheque certificado' in [x['description'] for x in i]) for i in dataframe.non_mercado_pago_payment_methods]
    #NLP features
    dataframe = build_NLP_features(dataframe)
    #Keep only necesary columns
    dataframe = dataframe[raw_columns]
    return dataframe

