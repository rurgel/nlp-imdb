from nltk import corpus
from tensorflow.keras.preprocessing.sequence import pad_sequences

def limpa_texto(df, *, coluna_texto='Descrição'):
    '''
     Entradas
        df: <pandas.DataFrame>
            Dataframe
        coluna_texto (opcional): <str>
            Nome da coluna com o texto a ser processado. (Default: 'Texto')
     Saídas:
        texto: <pandas.Series>
            Texto processado
    '''
    stopwords = corpus.stopwords.words('portuguese')
    stopwords.remove('não')

    texto = df[coluna_texto].str.lower()
    texto = texto.str.replace( "[^a-záàâãéêeíïóôõöúüç\':_]", " ", regex=True )
    texto = texto.apply( lambda x: ' '.join( [word for word in x.split() 
                                                    if word not in stopwords]))
    return texto

def gera_tokens(tokenizer, text, *, padding_len=None):
    '''
     Entradas
        tokenizer: <>

        text: <pandas.Series>
            Conjunto de texto a ser tokenizado
        padding_len (opcional): <int>
            Tamanho do preenchimento das sequencias de saída
     Saídas:
        tokens: <numpy.array>
            Texto tokenizado
    '''
    tokens = tokenizer.texts_to_sequences(text)
    if not padding_len:
        padding_len = max([len(x) for x in tokens])
    tokens = pad_sequences( tokens, 
                            maxlen=padding_len, 
                            padding='pre', 
                            truncating='post')
    return tokens