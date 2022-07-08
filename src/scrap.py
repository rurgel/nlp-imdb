#%% Bibliotecas necessárias
import requests 
from bs4 import BeautifulSoup 
import pandas as pd 
from tqdm import tqdm
import langid
from googletrans import Translator
import stanza
#%% Baixa informações do IMDB (demorado)
cols = ["Título", "Descrição", "Gêneros", "Nota"]
url_base = 'https://www.imdb.com/title/'
headers = {"Accept-Language": "pt-BR,pt;q=0.5"}
df = pd.read_csv('data/lista.csv')
filmes = pd.DataFrame(columns=cols)
for i,df_ in tqdm(df.iterrows()):
    url = url_base+df_['Filmes']
    try:
        with requests.get(url, headers=headers) as req:
            soup = BeautifulSoup(req.content, "html.parser")
        filme = [soup.find_all(
                        "h1", {"class": "sc-b73cd867-0 eKrKux"})[0].contents[0],
                 soup.find_all(
                        "span", {"class": "sc-16ede01-2 gXUyNh"})[0].contents[0],
                 [x.contents[0] for x in soup.find_all(
                        "li", {"class": "ipc-inline-list__item ipc-chip__text"})],
                 float(soup.find_all(
                        "span", {"class": "sc-7ab21ed2-1 jGRxWM"})[0]
                    .contents[0]
                    .replace(',','.'))
                ]
        filmes = filmes.append(pd.Series(filme, index=cols), ignore_index=True)
    except:
        pass
filmes['Gêneros']=filmes.apply(lambda row: ','.join(row['Gêneros']), axis=1)
filmes['Descrição'] = filmes['Descrição'].apply(lambda x: x.replace(';',','))
filmes.to_csv('data/raw/imdb.csv', index=False, encoding='utf-8-sig', sep=';')
#%% Traduz textos em espanhol/inglês
filmes = pd.read_csv('data/raw/imdb.csv', sep=';')
filmes.dropna(inplace=True)
langid.set_languages(['es','pt', 'en'])  # ISO 639-1 codes
def idioma(texto):
    lang, score = langid.classify(texto)
    return lang
filmes['Idioma do Texto'] = filmes['Descrição'].apply(idioma)
print(filmes['Idioma do Texto'].value_counts().to_string())

translator = Translator()
filmes.loc[filmes['Idioma do Texto']=='es','Descrição'] = filmes.loc[
           filmes['Idioma do Texto']=='es','Descrição']\
        .apply(lambda x: translator.translate(x, dest="pt").text)
filmes.loc[filmes['Idioma do Texto']=='en','Descrição'] = filmes.loc[
           filmes['Idioma do Texto']=='en','Descrição']\
        .apply(lambda x: translator.translate(x, dest="pt").text)

filmes.drop(columns=['Idioma do Texto'])\
      .to_csv('data/processed/imdb_traduzido.csv', 
              index=False, encoding='utf-8-sig', sep=';')
#%% Lemmatization
nlp = stanza.Pipeline('pt', processors='tokenize,mwt,pos,lemma')

def lemma(frase):
    doc = nlp(frase)
    return ' '.join([word.lemma for sent in doc.sentences for word in sent.words])
from tqdm import tqdm
filmes = pd.read_csv('data/processed/imdb_traduzido.csv', sep=';')
filmes.rename(columns={'Descrição':'Descrição IMDB'}, inplace=True)
for index, row in tqdm(filmes.iterrows(), total=filmes.shape[0]):
    filmes.at[index,'Descrição']=lemma(row['Descrição IMDB']).replace(' .','.').replace(' ,',',')

filmes.to_csv('data/processed/imdb_pt_lemma.csv', 
              index=False, encoding='utf-8-sig', sep=';')