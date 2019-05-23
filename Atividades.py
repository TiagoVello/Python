import pandas as pd

nomes = ['Pinduca',
         'Prec',
         'Ed',
         'Mosko',
         'Foquinha',
         'Kuri',
         'Shao',
         'Palestrinha',
         'Migo',
         'Mingau',
         'Dorinha',
         'Buzz',
         'Cidão',
         'Dolfinho',
         'Shaozinho',
         'Didi']
         
atividades = [['Colocar ração para os cachorros e para as gatas','4x por dia para as gatas e 2x por dia para os cachorros'],
              ['Colocar ração para os cachorros e para as gatas','4x por dia para as gatas e 2x por dia para os cachorros'],
              ['Fazer gelo e deixar a garrafa de agua cheia','diária'],
              ['Lixos DO BANHEIRO COMUM E DA COZINHA (repor saco de lixo)','diária'],
              ['Passear com os dogs','2x por semana'],
              ['Passear com os dogs','2x por semana'],
              ['Varrer area comum','2x por semana'],
              ['Varrer area comum','2x por semana'],
              ['Lavar lixos','Semanal'],
              ['Assuntos Pontuais','---------'],
              ['Recolher louça suja segunda feira','1x por semana'],
              ['Guardar louça limpa e seca','diária'],
              ['Limpar a areia','2x por dia'],
              ['Pegar as merda dos dogs','2x por semana'],
              ['Passar a redinha','diária'],
              ['Limpar a geladeira. Tirar panelas e coisas estragadas','semanal']]
    
def exibir(atividades, nomes):
    print('LISTA DE ATIVIDADES')
    print('')
    for i in range(len(nomes)):
        print('Parabéns, {} foi presenteado com: {} de frequência {}'.format(nomes[i],atividades[i][0],atividades[i][1]))
        print('')
        
def proxima_atividade(atividades):
    salvar_atividades = atividades[-1]
    for i in range(len(nomes)-1):
        atividades[-1-i] = atividades[-2-i]
    atividades[0] = salvar_atividades
    return atividades

proxima_atividade(atividades)
exibir(atividades,nomes)

print(len(atividades))
print(len(nomes))
    