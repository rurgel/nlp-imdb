import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def target( filmes, *, genero=None, drop=None ):
    '''
     Entradas
        filmes: <pandas.DataFrame>
            Dataframe com os filmes
        genero (opcional): <str> ou <list>
            Gênero dos filmes a serem selecionados
        drop (opcional):  <str> ou <list>
            Gênero dos filmes a serem ignorados
     Saídas:
        Y: <np.array>
            Dataframe convertido
        categorias: <list>
            Gêneros identificados
    '''
    filmes.index.name = 'id'
    filmes=filmes.copy()
    filmes['Gênero'] = filmes['Gênero'].apply(lambda x: x.split(','))
    tipo = filmes.explode('Gênero')
    ohe = OneHotEncoder( sparse=False, handle_unknown='ignore' )
    ohe.fit(tipo[['Gênero']])
    Y = ohe.fit_transform(tipo[['Gênero']])
    Y = pd.DataFrame( Y, index=tipo.index, columns=ohe.categories_[0] )
    Y = Y.groupby('id').max()
    if drop:
        if isinstance(drop, str):
            drop = [drop]
        try:
            Y.drop(columns=drop, inplace=True)
        except:
            pass
    if genero:
        if isinstance(genero, str):
            genero = [genero]
        Y = Y.loc[:,genero]
    categorias = list(Y.columns)
    Y = Y.to_numpy()
    return Y, categorias