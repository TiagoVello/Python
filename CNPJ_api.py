import pandas
import requests
import json
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pandas.io.json import json_normalize
pd.set_option('display.max_columns', 30)

# importando para uma variavel (DataFrame)
tabela_cnpj = pandas.read_excel(r'Dados/dados_cnpj_filtrados.xlsx')

# vizualizando os dados
tabela_cnpj.columns
tabela_cnpj['file']
tabela_cnpj['tipo']
tabela_cnpj['cnpj']

# abrindo o navegador
browser = webdriver.Chrome('chromedriver.exe')
browser.implicitly_wait(2)
receita
receita = []
dados = DataFrame()
for i, cnpj in enumerate(tabela_cnpj['cnpj']):
    api_url = r'https://www.receitaws.com.br/v1/cnpj/{}'.format(cnpj)
    browser.get(api_url)
    body = browser.find_element_by_tag_name('body').text
    if ('Too many requests, please try again later.' not in body):
        item = browser.find_element_by_tag_name('pre').text
        parsed_item = json.loads(item)
        data_frame_item = json_normalize(parsed_item)
        receita.append(data_frame_item)
    if (i % 3) == 0:
        time.sleep(61)
    if i == 90:
        break
        

# teste
teste_url = r'https://www.receitaws.com.br/v1/cnpj/28985101000188'
r = requests.get(teste_url)
b = BeautifulSoup(r.text, 'xml')
b.find('style').get_text()  
    
