def soma_espaços(numero_espaços):
    espaços = ''
    for e in range(numero_espaços):
        espaços += ' '
    return espaços

def soma_digitos(numero_digitos):
    digitos = ''
    for d in range(numero_digitos):
        digitos += str(d)
    return digitos

def soma_digitos_invertidos(numero_digitos):
    digitos = ''
    for d in range(numero_digitos-2,-1,-1):
        digitos += str(d)
    return digitos

def TriânguloEsquerdo(tamanho):
    for linha in range(2*tamanho):
        espaços = soma_espaços(abs(tamanho-linha))
        digitos = soma_digitos(tamanho-abs(linha-tamanho))
        print(espaços+digitos)

def Losangulo(tamanho):
    for linha in range(2*tamanho):
        espaços = soma_espaços(abs(tamanho-linha))
        digitos = soma_digitos(tamanho-abs(linha-tamanho))
        digitos_invertidos = soma_digitos_invertidos(tamanho-abs(linha-tamanho))
        print(espaços+digitos+digitos_invertidos)
        
def TriânguloDireito(tamanho):
    for linha in range(2*tamanho):
        digitos_invertidos = soma_digitos_invertidos(tamanho-abs(linha-tamanho))
        print(digitos_invertidos)
        
def TriânguloSuperior(tamanho):
    for linha in range(tamanho):
        espaços = soma_espaços(abs(tamanho-linha))
        digitos = soma_digitos(tamanho-abs(linha-tamanho))
        digitos_invertidos = soma_digitos_invertidos(tamanho-abs(linha-tamanho))
        print(espaços+digitos+digitos_invertidos)

def TriânguloInferior(tamanho):
    for linha in range(tamanho,tamanho*2):
        espaços = soma_espaços(abs(tamanho-linha))
        digitos = soma_digitos(tamanho-abs(linha-tamanho))
        digitos_invertidos = soma_digitos_invertidos(tamanho-abs(linha-tamanho))
        print(espaços+digitos+digitos_invertidos)
     
f = input('Digite a figura : ')
t = int(input('Digite o tamanho da figura : '))
if f == 'TE':
    TriânguloEsquerdo(t)
elif f == 'L':
    Losangulo(t)

#Rascunho
TriânguloEsquerdo(10)       
TriânguloDireito(10)
TriânguloSuperior(10)
TriânguloInferior(10)
Losangulo(10)
        
a = ''
for i in range(10):
    a += str(i)
print(a)
    print(soma_espaços(10)+soma_digitos(10))