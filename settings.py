from nltk.corpus import stopwords

hyper_parameters = {
                    'max_depth': 35,
                    'min_samples_split': 20,
                    'n_estimators': 110,
                    'random_state': 1
                    }

# NLP Features Auxiliar
keywords_list = [['nuevo', 'nueva', 'nuevos', 'nuevas'], ['usado', 'usada', 'usados', 'usadas', 'uso', 'usa'],
                 ['caja'], ['canje', 'canjes'], ['permuto'], ['venta'], ['vendo'], ['gratis'], ['restaurado',
                 'restaurada', 'restaurados', 'restauradas'], ['original','originales'], ['antiguo', 'antigüo',
                 'antigua', 'antigüa'], ['excelente'], ['buen'], ['impecable'], ['estado'], ['condicion',
                 'condición', 'condiciones'], ['garantia'], ['importada', 'importado'], ['fallas', 'falla'],
                 ['oficial', 'oficiales'], ['tienda'],['store'], ['promo', 'oferta'], ['outlet'], ['renovado',
                 'renovada', 'renovados', 'renovadas'], ['historia'], ['coleccion'], ['retro', 'completo'],
                 ['universal'], ['cerrados'], ['lote'], ['ideal']]

stopwords_spanish = stopwords.words('spanish')


# List of columns to be pre-processed 
numeric_columns = ['PRICE', 'QTY_INITIAL', 'QTY_AVAILABLE', 'QTY_SOLD', 'SHARE_SOLD', 'QTY_VARIATIONS',
                    'LEN_ATRIBUTOS', 'LATITUDE', 'LONGITUDE', 'CREATION_TO_UPDATE', 'CREATION_TO_STOP',
                    'TITLE_VEC_0', 'TITLE_VEC_1', 'TITLE_VEC_2', 'TITLE_VEC_3', 'TITLE_VEC_4', 'TITLE_VEC_5',
                    'TITLE_VEC_6', 'TITLE_VEC_7', 'TITLE_VEC_8', 'TITLE_VEC_9', 'TITLE_VEC_10', 'TITLE_VEC_11',
                    'TITLE_VEC_12', 'TITLE_VEC_13', 'TITLE_VEC_14', 'TITLE_VEC_15', 'TITLE_VEC_16', 'TITLE_VEC_17',
                    'TITLE_VEC_18', 'TITLE_VEC_19', 'TITLE_VEC_20', 'TITLE_VEC_21', 'TITLE_VEC_22', 'TITLE_VEC_23',
                    'TITLE_VEC_24', 'TITLE_VEC_25', 'TITLE_VEC_26', 'TITLE_VEC_27', 'TITLE_VEC_28', 'TITLE_VEC_29',
                    'TITLE_VEC_30', 'TITLE_VEC_31', 'TITLE_VEC_32', 'TITLE_VEC_33', 'TITLE_VEC_34', 'TITLE_VEC_35',
                    'TITLE_VEC_36', 'TITLE_VEC_37', 'TITLE_VEC_38', 'TITLE_VEC_39', 'TITLE_VEC_40', 'TITLE_VEC_41',
                    'TITLE_VEC_42', 'TITLE_VEC_43', 'TITLE_VEC_44', 'TITLE_VEC_45', 'TITLE_VEC_46', 'TITLE_VEC_47',
                    'TITLE_VEC_48', 'TITLE_VEC_49', 'TITLE_VEC_50', 'TITLE_VEC_51', 'TITLE_VEC_52', 'TITLE_VEC_53',
                    'TITLE_VEC_54', 'TITLE_VEC_55', 'TITLE_VEC_56', 'TITLE_VEC_57', 'TITLE_VEC_58', 'TITLE_VEC_59',
                    'TITLE_VEC_60', 'TITLE_VEC_61', 'TITLE_VEC_62', 'TITLE_VEC_63', 'TITLE_VEC_64', 'TITLE_VEC_65',
                    'TITLE_VEC_66', 'TITLE_VEC_67', 'TITLE_VEC_68', 'TITLE_VEC_69', 'TITLE_VEC_70', 'TITLE_VEC_71',
                    'TITLE_VEC_72', 'TITLE_VEC_73', 'TITLE_VEC_74', 'TITLE_VEC_75','TITLE_VEC_76', 'TITLE_VEC_77',
                    'TITLE_VEC_78', 'TITLE_VEC_79', 'TITLE_VEC_80', 'TITLE_VEC_81', 'TITLE_VEC_82','TITLE_VEC_83',
                    'TITLE_VEC_84', 'TITLE_VEC_85', 'TITLE_VEC_86', 'TITLE_VEC_87', 'TITLE_VEC_88', 'TITLE_VEC_89',
                    'TITLE_VEC_90', 'TITLE_VEC_91', 'TITLE_VEC_92', 'TITLE_VEC_93', 'TITLE_VEC_94', 'TITLE_VEC_95']

binary_columns = ['FLAG_MERCADOPAGO', 'FLAG_AUTO_RELIST', 'FLAG_BIDS_VISITS', 'FLAG_VISITS', 'FLAG_GOOD_THUMBNAIL',
                  'FLAG_POOR_THUMBNAIL', 'FLAG_FREE_RELIST', 'FLAG_VARIATIONS', 'FLAG_GARANTIA', 'FLAG_CONTACT',
                  'FLAG_EMAIL', 'FLAG_LOCAL_PICK_UP', 'FLAG_FREE_SHIPPING', 'ACCEPTS_EFECTIVO',
                  'ACCEPTS_TRANSFERENCIA','ACCEPTS_TARJETA', 'ACCEPTS_ACORDAR', 'ACCEPTS_GIRO', 'ACCEPTS_MP',
                  'ACCEPTS_VISA', 'ACCEPTS_MASTER','ACCEPTS_REEMBOLSO', 'ACCEPTS_VISA_ELECTRON', 'ACCEPTS_MAESTRO',
                  'ACCEPTS_AMERICAN', 'ACCEPTS_DINERS','ACCEPTS_CHEQUE', 'FLAG_LOCATION', 'FLAG_OPEN_HOURS', 'NUEVO',
                  'USADO', 'CAJA', 'CANJE', 'PERMUTO','VENTA', 'VENDO', 'GRATIS', 'RESTAURADO', 'ORIGINAL', 'ANTIGUO',
                  'EXCELENTE', 'BUEN', 'IMPECABLE', 'ESTADO', 'CONDICION', 'GARANTIA', 'IMPORTADA', 'FALLAS',
                  'OFICIAL', 'TIENDA', 'STORE', 'PROMO', 'OUTLET', 'RENOVADO', 'HISTORIA', 'COLECCION', 'RETRO',
                  'UNIVERSAL', 'CERRADOS', 'LOTE', 'IDEAL']

categoric_columns = ['TAG_SUSPEND_STATUS', 'TAG_BUY_MODE', 'TAG_LISTING']

fill_nan_columns = ['QTY_VARIATIONS', 'LEN_ATRIBUTOS', 'LATITUDE', 'LONGITUDE','FLAG_BIDS_VISITS', 'FLAG_VISITS',
                    'FLAG_GOOD_THUMBNAIL', 'FLAG_POOR_THUMBNAIL', 'FLAG_FREE_RELIST', 'FLAG_OPEN_HOURS']

raw_columns = numeric_columns + binary_columns + categoric_columns


# Input features for the model
categoric_to_binary_columns = ['TAG_SUSPEND_STATUS_deleted', 'TAG_SUSPEND_STATUS_expired', 'TAG_SUSPEND_STATUS_no_value',
                              'TAG_SUSPEND_STATUS_suspended', 'TAG_SUSPEND_STATUS_nan', 'TAG_BUY_MODE_auction',
                              'TAG_BUY_MODE_buy_it_now', 'TAG_BUY_MODE_classified', 'TAG_BUY_MODE_nan', 'TAG_LISTING_bronze',
                              'TAG_LISTING_free', 'TAG_LISTING_gold', 'TAG_LISTING_gold_premium', 'TAG_LISTING_gold_pro',
                              'TAG_LISTING_gold_special', 'TAG_LISTING_silver', 'TAG_LISTING_nan']

features = numeric_columns + binary_columns + categoric_to_binary_columns

            