# Coleção de funções para a disciplina de OU3
import numpy as np
from scipy.optimize import fsolve
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def f_Pvap_Antoine_db(Temp, i_comp, dados):
    #import numpy as np
    ''' Função que calcula a pressão de vapor, segundo a equação de Antoine, para o componente
      i_comp presente no databank_properties.pickle.
      Equação de Antoine: Pvap = exp(A - B /(Temp + C)), com: 
      [Temp] = K
      [Pvap] = mmHg
      Entrada (argumentos da função)
      Temp   = temperatura em K para a qual será calculada a Pvap
      i_comp = inteiro que corresponde ao número do componente no banco de dados
      dados  = pandas dataframe com os dados lidos do arquivo
      Saida: tupla
      Pvap - pressão de vapor do i_comp em mmHg
      par = dicionário com os parâmetros A, B e C da equação de Antoine
    '''
    # param <- as.numeric(param)
    par_array = np.array(dados[dados['num'] == i_comp][['pvap_a','pvap_b','pvap_c']])[0]
    par = {'a': par_array[0], 'b': par_array[1], 'c': par_array[2]}
    a = par['a']
    b = par['b']
    c = par['c']
    Pvap = np.exp(a - b/(Temp + c))
    # attr(x = Pvap, which = "units") <- "mmHg"
    return Pvap, par

def f_K_Raoult_db(T_eq, P_eq, lista_componentes, dados):
    # import numpy as np
    ''' Função para o cálculo da volatilidade segundo a Lei de Raoult:
        - fase vapor -> mistura de gás ideal
        - fase líquida -> solução normal
        K = P_vap(Teq) / P_eq
        Entrada (argumentos da função)
        T_eq - temperatura de equilíbrio em K
        P_eq - pressão de equilíbrio em mmHg
        lista_componentes - lista com os números inteiro dos componentes no databank
        dados - pandas dataframe com os dados do databank_properties.pickle
        Saida: tupla
        K_comp - np.array com os valores da volatilidade na ordem da lista_componentes
        P_vap_comp - np.array com os valores de P_vap segundo a equação de Antoine e os parâmetros
                    do databank_properties.pickle
    '''
    nc = len(lista_componentes)
    P_vap_comp = np.empty(nc)
    K_comp = np.empty(nc)
    k = 0
    for i_comp in lista_componentes:
        P_vap_comp[k], par = f_Pvap_Antoine_db(T_eq, i_comp, dados)
        K_comp[k] = P_vap_comp[k] / P_eq
        k += 1
    return K_comp, P_vap_comp

def f_Pb_T(Temp,P,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Temperatura do ponto de bolha
      Entrada:
      Temp - temperaura de equilíbrio em K - variável implícita da equação
      P - pressão de equilíbrio em mmHg
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(Temp) == float):
    Temp = np.array([Temp])
  nc = len(z)
  nr = len(Temp)
  MP = np.empty((nr,nc))
  x = z
  for i, T_vez in enumerate(Temp):
    K_comp = f_K_Raoult_db(T_vez, P, lista_componentes, dados)[0]
    MP[i,:] = K_comp * x
  f = 1 - np.sum(MP, axis=1)
  return f

def f_Po_T(Temp,P,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Temperatura do ponto de orvalho
      Entrada:
      Temp - temperaura de equilíbrio em K - variável implícita da equação
      P - pressão de equilíbrio em mmHg
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(Temp) == float):
    Temp = np.array([Temp])
  nc = len(z)
  nr = len(Temp)
  MP = np.empty((nr,nc))
  y = z
  for i, T_vez in enumerate(Temp):
    K_comp = f_K_Raoult_db(T_vez, P, lista_componentes, dados)[0]
    MP[i,:] = y / K_comp
  f = 1 - np.sum(MP, axis=1)
  return f

def f_Pb_P(P,Temp,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Pressão do ponto de bolha
      Entrada:
      P - pressão de equilíbrio em mmHg - variável implícita da equação
      Temp - temperaura de equilíbrio em K 
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(P) == float):
    P = np.array([P])
  nc = len(z)
  nr = len(P)
  MP = np.empty((nr,nc))
  x = z
  for i, P_vez in enumerate(P):
    K_comp = f_K_Raoult_db(Temp, P_vez, lista_componentes, dados)[0]
    MP[i,:] = K_comp * x
  f = 1 - np.sum(MP, axis=1)
  return f

def f_Po_P(P,Temp,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Pressão do ponto de orvalho
      Entrada:
      P - pressão de equilíbrio em mmHg - variável implícita da equação
      Temp - temperaura de equilíbrio em K 
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(P) == float):
    P = np.array([P])
  nc = len(z)
  nr = len(P)
  MP = np.empty((nr,nc))
  y = z
  for i, P_vez in enumerate(P):
    K_comp = f_K_Raoult_db(Temp, P_vez, lista_componentes, dados)[0]
    MP[i,:] = y / K_comp
  f = 1 - np.sum(MP, axis=1)
  return f

def f_calculo_PbPo_db(vp, x_pot, z, lista_componentes, dados):
    ''' Função para o cálculo das temperatura ou pressões do ponto de bolha 
          e do ponto de orvalho ( [T] em K e [P] em mmHg)
        Entradas:
        vp - variável do problema 'T' ou 'P' - string
        x_pot - valor de pressão ou temperatura dado
        z - composição da carga em fração molar
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
        dados - dataframe com os dados do databank
        Saidas:
        Se vp == 'T' -> T_Pb, T_Po, T_eb_comp = lista com as temperaturas de
                        ebulição normal dos componentes
        Se vp == 'P' -> P_Pb, P_Po, M_P_vap = matriz com as pressões de 
                        vapor dos componentes nas T_eb_comp
    '''
    #from scipy.optimize import fsolve
    nc = len(lista_componentes)
    if (vp == 'T'):
        P_eq = x_pot
        T_eb_comp = dados[dados['num'].isin(lista_componentes)]['boiling_point']
        T_eb_comp = T_eb_comp.tolist()
        T_guest = (min(T_eb_comp) + max(T_eb_comp) )/2
        T_Pb = fsolve(f_Pb_T, T_guest, args=(P_eq, z, lista_componentes, dados))[0]
        T_Po = fsolve(f_Po_T, T_guest, args=(P_eq, z, lista_componentes, dados))[0]
        return (T_Pb, T_Po, T_eb_comp)
    if (vp == 'P'):
        T_eq = x_pot
        T_eb_comp = dados[dados['num'].isin(lista_componentes)]['boiling_point']
        T_eb_comp = T_eb_comp.tolist()
        P_vap_eb_comp = np.empty(nc)
        k = 0
        for i_comp in lista_componentes:
          P_vap_eb_comp[k] = f_Pvap_Antoine_db(T_eq, i_comp, dados)[0]
          k += 1
        P_guest = (np.min(P_vap_eb_comp) + np.max(P_vap_eb_comp))/2
        P_Pb = fsolve(f_Pb_P, P_guest, args=(T_eq, z, lista_componentes, dados))[0]
        P_Po = fsolve(f_Po_P, P_guest, args=(T_eq, z, lista_componentes, dados))[0]
        return (P_Pb, P_Po, P_vap_eb_comp)

def f_res_RR_flash_db(fv, z, P, Temp, lista_componentes, dados):
    ''' Função que determina o resíduo da equação de Rachford-Rice para o flash
        multicomponente na solução para encontrar fv (fração vaporizada da carga)
      Entrada:
      fv - fração vaporizada da carga - variável implícita
      z - composição da carga em fração molar
      P - pressão do flash em mmHg
      T - temperatura do flash em K
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saidas:
      res - resíduo na busca da solução - res = 0 -> solução
    '''
    nc = len(lista_componentes)
    if (type(fv) == float):
      fv = np.array([fv])
    nr = len(fv)
    K_comp = f_K_Raoult_db(Temp, P, lista_componentes, dados)[0]
    M_parc = np.empty((nr, nc))
    num = z * K_comp
    for i, fv_vez in enumerate(fv):
        den = 1.0 + fv_vez*(K_comp - 1.0)
        M_parc[i,:] = num / den
    res = 1.0 - np.sum(M_parc, axis=1)
    return res

def f_sol_RR_flash_db(z, P, Temp, lista_componentes, dados):
    ''' Função que resolve a equação de Rachford-Rice e encontra a fv 
        (fração vaporizada da carga)
        Entrada:
        z - composição da carga em fração molar
        P - pressão do flash em mmHg
        T - temperatura do flash em K
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
        dados - dataframe com os dados do databank
        Saidas: {dicionário}
        fv_flash - fração vaporizada - solução do flash
        x_eq - composição do líquido no equilíbrio
        y_eq - composição do vapor no equilíbrio
        K_comp - volatilidade dos componentes
        alpha_comp - volatilidade relativa em relação ao componente chave pesado (i_chk)
    '''
    fv_guest = 0.5
    fv_flash = fsolve(f_res_RR_flash_db, fv_guest, args=(z, P, Temp, lista_componentes, dados))[0]
    K_comp = f_K_Raoult_db(Temp, P, lista_componentes, dados)[0]
    num = z * K_comp
    den = 1.0 + fv_flash*(K_comp - 1.0)
    y_eq = num / den
    x_eq = y_eq / K_comp
    i_chk = np.argmin(K_comp)
    alpha_comp = K_comp/K_comp[i_chk]
    return {'fv_flash': fv_flash, 'x_eq': x_eq, 'y_eq': y_eq, 'K_comp': K_comp,
            'alpha_comp':alpha_comp}

def f_sol_ELV_2c_db(Temp, P, lista_componentes, dados):
    ''' Função para o cálculo do ELV em um sistema binário ideal (Lei de Raoult)
      Entrada:
      P - pressão do flash em mmHg
      T - temperatura do flash em K
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saidas: tupla de vetores
      x_eq - concentrações do componentes no ELV na fase líquida
      y_eq - concentrações do componentes no ELV na fase vapor
    '''
    nc = len(lista_componentes)
    P_vap_comp = f_K_Raoult_db(Temp, P, lista_componentes, dados)[1]
    v_rhs = np.array([1,1,0,0])
    A_elv = np.array([[1,1,0,0],
                      [0,0,1,1],
                      [P_vap_comp[0],0,-P,0],
                      [0, P_vap_comp[1], 0, -P]])
    x_sol = np.linalg.inv(A_elv) @ v_rhs
    x_eq = np.empty(nc)
    y_eq = np.empty(nc)
    x_eq[0] = x_sol[0]
    x_eq[1] = x_sol[1]
    y_eq[0] = x_sol[2]
    y_eq[1] = x_sol[3]
    return (x_eq, y_eq)

def f_gerar_dados_elv_2c_bd(P_eq, n_pontos, lista_componentes, dados):
    ''' Função para gerar um pandas.dataframe com n_pontos instâncias de dados
          do ELV de um sistema binário ideal
        Entradas:
        P_eq - pressão de equilíbrio em mmHg
        n_pontos - número de instâncias geradas
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
        dados - dataframe com os dados do databank
        Saida: pandas.dataframe
        dados_elv - com as seguintes series: 'T', 'x1' e 'y1'
    '''
    T_eb_comp = dados[dados['num'].isin(lista_componentes)]['boiling_point']
    T_eb_comp = T_eb_comp.tolist()
    T_faixa = np.linspace(T_eb_comp[0], T_eb_comp[1], n_pontos)
    dados_elv = pd.DataFrame({'T': T_faixa})
    for i, T in enumerate(dados_elv['T']):
        x_eq, y_eq = f_sol_ELV_2c_db(T, P_eq, lista_componentes, dados)
        dados_elv.loc[i,'x1'] = x_eq[0]
        dados_elv.loc[i,'y1'] = y_eq[0]
    return dados_elv

def f_reta_flash(x1, z1, fv):
  ''' Função da reta deoperação do flash no diagrma y-x
      
      Entrada:

      x1 = concentração do componente mais volátil na fase líquida
      
      z1 = concentração do componente mais volátil na carga (F)
      
      fv = fração vaporizada
      
      Saida:
      
      y1 = composição do componente mais volátil na fase vapor pertencente
           a reta de operação de flash
  '''
  if (fv == 0):
    n_pontos = len(x1)
    y1 = np.linspace(z1,1.0,n_pontos)
  else:
    a = -(1.0 - fv)/ fv
    b = z1/fv
    y1 = a*x1 + b
  return y1

def f_gera_diag_y_x(P_eq, n_pontos, lista_componentes, dados):
    ''' Função que gera o diagrama y-x de uma mistura binária
        
        P_eq - pressão de equilíbrio em mmHg
    '''
    dados_elv = f_gerar_dados_elv_2c_bd(P_eq, n_pontos, lista_componentes, dados)
    # Fazendo o gráfico
    fig1, ax1 = plt.subplots(figsize =(4,4))
    ax1.plot(dados_elv['x1'], dados_elv['y1'], 'b', label='Equilíbrio')
    ax1.plot(dados_elv['x1'], dados_elv['x1'], 'k', label= r'$y_1 = x_1$')
    # Adicionando texto nos eixos - descrição
    ax1.set_xlabel('x1 - fase líquida')
    ax1.set_ylabel('y1 - fase vapor')
    # Adicionando título para a figura
    ax1.set_title('Diagrama y-x')
    # Adicionando um texto
    ax1.text(0.4, 1.0, r'@$P_{eq} = 760.0 \, mmHg$')
    # Adicionando uma legenda
    ax1.legend()
    # Adicionando linha vertical
    # ax1.vlines(z[0],T_Pb, T_Po, colors='m', linestyles='dashed')
    #ax1.axvline(sol_flash['x_eq'][0])
    #ax1.axvline(sol_flash['y_eq'][0])
    # Adicionando linha horizontal
    #ax1.hlines(T_flash,0, 1, colors='y', linestyles='solid')
    # Adicionando grade
    ax1.grid()
    # Ajustando os ticks
    from matplotlib.ticker import (MultipleLocator) #AutoMinorLocator
    ax1.xaxis.set_major_locator(MultipleLocator(0.10))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax1.yaxis.set_major_locator(MultipleLocator(0.10))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax1.tick_params(which='minor', length=4, color='r')
    #ax1.grid(True, which='minor')
    plt.show()
    return fig1, ax1


def f_adi_reta_flash(fig_f, ax_f, z, T_flash, P_flash, lista_componentes, dados, fv=float('NaN')):
    ''' Adiciona uma reta de flash no diagrama y-x'''
    import math
    # Verificação e cálculo da fração vaporizada da carga --> fv
    if (math.isnan(fv)):
        sol_flash = f_sol_RR_flash_db(z, P_flash, T_flash, lista_componentes, dados)
        fv = sol_flash['fv_flash']
    # determinação do menor valor em x para a reta de operação do flash
    if (fv == 0.0):
        x_min_fv = z[0]
    elif (fv == 1.0):
        x_min_fv = 0.0
    else:
        x_min_fv =(fv-z[0])/(fv-1)
        if (x_min_fv < 0):
            x_min_fv = 0.0
        x_eq_1 = sol_flash['x_eq'][0]
        y_eq_1 = sol_flash['y_eq'][0]
        ax_f.axvline(x_eq_1, linestyle='dotted')
        ax_f.axhline(y_eq_1, linestyle='dotted')
    # Geração dos pontos da reta de operação do flash
    x1_graf = np.linspace(x_min_fv,z[0],20)
    y1_rf_graf = f_reta_flash(x1_graf,z[0], fv)
    # inserção da reta de flash do diagrama y-x
    ax_f.plot(x1_graf, y1_rf_graf, 'r', label= 'reta do flash')
    ax_f.vlines(z[0],0, z[0], colors='m', linestyles='dashed')
    ax_f.legend()
    plt.show()
    return fig_f, ax_f

def f_cp_vap(Temp,param):
    '''Função que calcula cp do vapor como  gás ideal para os dados do 
        databank_properties.pickle
    '''
    a = param[0]
    b = param[1]
    c = param[2]
    d = param[3]
    cp = a + b*Temp + c*Temp**2 + d*Temp**3
    # attr(x = cp, which = "units") <- "cal/mol_K"
    cp = 4.184 * cp # conversão de cal para J (Joules)
    #cp = 1000.0 * cp # conversão de mol para kmol
    return cp # J/mol/K


def f_cp_param(T1, lista_componentes, dados):
    '''Função que monta uma matriz com os parâmetros de todos os compoenente da 
        lista_componentes e também calcula o valor de cp de cada um deles na 
        temperatura T1
        Entradas:
        T1 = temperatura em K
        lista_componentes =
        dados =
        Saidas:
        v_cp = vetor com os valores de cp @T1 em cal/mol/K
        M_param = matriz com os quatro parâmetrso da equação do modelo de cp, sendo um
                  componente por linha na mesma ordem de lista_componentes
        '''
    nc = len(lista_componentes)
    M_param = np.empty((nc,4))
    v_cp = np.empty((nc))
    k = 0
    for i_num in lista_componentes:
        #print(i_num)
        param = dados [dados['num'] == i_num][['cp_a', 'cp_b', 'cp_c', 'cp_d']]
        param = param.to_numpy()[0]
        M_param[k,:] = param
        v_cp[k] = f_cp_vap(T1,param)
        k += 1
    # v_cp em J/mol/K
    return (v_cp, M_param)

def f_H_vap_ig_stream(y_stream, T_ref, Temp, lista_componentes, dados):
    '''Função para o cálculo para aentalpia de uma corrente na fase vapor e 
        considerada como gás ideal
        Entradas:
        y_stream = 
        T_ref = 
        Temp = 
        lista_componentes =
        dados =
        Saídas: em uma tupla
        H_stream = entalpia da corrente com composição y_stream
        DH = DeltaH dos componente de T_ref até Temp
    '''
    nc = len(lista_componentes)
    M_param = f_cp_param(T_ref, lista_componentes, dados)[1]
    DH = np.empty((nc,))
    for i in range(nc):
        DH[i] = integrate.quad(f_cp_vap, T_ref, Temp, args =(M_param[i,:],))[0]
    H_stream = y_stream @ DH
    # Entalpias em J/mol ou kJ/kmol
    return (H_stream, DH)

def f_DHvap_ClausiusClayperon_db(Temp, lista_componentes, dados):
    ''' Função para o cálculo da entalpia e vaporrização a partir de um modelo
        de pressão de vapor (modelo de Antoine)
        Entradas:
        Temp = temperatura na qual deseja-se o valor da entalpia de vaporização
        lista_componentes =
        dados =
        Saidas:
        DH_vap_comp_T = valor da entalpia de vaporização na temperatura Temp em K para
                        todos os componentes da lista_componentes em J/mol
    '''
    nc = len(lista_componentes)
    R = 1.987207 # cal/mol.K
    T1 = Temp - 10.0
    T2 = Temp + 10.0
    tt = (1/T1) - (1/T2)
    DH_vap_comp_T = np.zeros((nc,))
    k = 0
    for i_comp in lista_componentes:
        Pv1 = f_Pvap_Antoine_db(T1, i_comp, dados)[0]
        Pv2 = f_Pvap_Antoine_db(T2, i_comp, dados)[0]
        DH_vap_comp_T[k] = R * np.log(Pv2/Pv1) / tt
        k += 1
    return DH_vap_comp_T * 4.184


def f_DHvap_Watson_db(Temp, lista_componentes, dados):
    ''' Função para o cálculo da entalpia de vaporização em função da temperatura
            a partir da entalpia de vaporização medida no ponto de ebulição normal.
            Utiliza o modelo de Watson que corresponde a eq.4.13 da p. 100 do SVNA.
        Entradas:
          Temp = temperatura em K
          lista_componentes = 
          dados = 
        Saídas:
          DH_vap_comp = vetor com as entalpias em Temp em J/mol
    '''
    nc = len(lista_componentes)
    T_BP_comp = dados[dados['num'].isin(lista_componentes)]['boiling_point'].to_numpy()
    DH_vap_comp_bp = dados[dados['num'].isin(lista_componentes)]['delta_h_vap_bp'].to_numpy()
    Tc_comp = dados[dados['num'].isin(lista_componentes)]['critical_temp'].to_numpy()
    Tr_eb_comp = T_BP_comp / Tc_comp # Temperatura de ebulição reduzida
    DH_vap_comp_T = np.zeros((nc,))
    for i in range(nc):
        frac = (1.0 - (Temp/Tc_comp[i]))/(1.0 - (Tr_eb_comp[i]))
        DH_vap_comp_T[i] = DH_vap_comp_bp[i]*((frac)**(0.38))
    return DH_vap_comp_T * 4.184


def f_gera_mod_locais_ps(dados_ps):
    ''' Modelo locais do diagrama entalpia composição para usar no método
            Ponchon-Savarit
            Entrada:
            dados_elv = dataframe do pandas com os dados de:
                T = temperatura em K
                x1 = composição de equilíbrio na fase líquida (liquido saturado)
                y1 = composição de equilíbrio na fase vapor (vapor saturado)
                DH_vap_Watson = entalpia de vaporização da mistura, composição do vapor,
                                calculada com o modelo de Watson
                Hig_v = entalpia do vapor saturado (J/mol)
                Hig_l = entalpia do líquido saturado (J/mol)
            Saidas:
            modelos = lista com os objetos dos respecitovos modelos na ordem:
                      mod_H_x, mod_H_y, mod_y_x, mod_x_y
            r2_modelos = lista com os valores dos coeficientes de determinação dos
                         modelos estimados
    '''
    # Modelo local para Hig_l = f(x1)
    x_regr = dados_ps['x1'].to_numpy().reshape(-1,1)
    y_regr = dados_ps['Hig_l'].to_numpy()
    polinomio_2g = PolynomialFeatures(degree = 2)
    X_poli_2g = polinomio_2g.fit_transform(x_regr)
    mod_H_x = LinearRegression()
    mod_H_x.fit(X_poli_2g, y_regr)
    r2_H_x = r2_score(y_regr, mod_H_x.predict(X_poli_2g))
    # Modelo local para Hig_v = f(y1)
    x_regr = dados_ps['y1'].to_numpy().reshape(-1,1)
    y_regr = dados_ps['Hig_v'].to_numpy()
    polinomio_2g = PolynomialFeatures(degree = 2)
    X_poli_2g = polinomio_2g.fit_transform(x_regr)
    mod_H_y = LinearRegression()
    mod_H_y.fit(X_poli_2g, y_regr)
    r2_H_y = r2_score(y_regr, mod_H_y.predict(X_poli_2g))
    # Modelo local para y1 = f(x1) - 3º grau
    x_regr = dados_ps['x1'].to_numpy().reshape(-1,1)
    y_regr = dados_ps['y1'].to_numpy()
    polinomio_3g = PolynomialFeatures(degree = 3)
    X_poli_3g = polinomio_3g.fit_transform(x_regr)
    mod_y_x = LinearRegression()
    mod_y_x.fit(X_poli_3g, y_regr)
    r2_y_x = r2_score(y_regr, mod_y_x.predict(X_poli_3g))
    # Modelo local para x1 = f(y1) - 3º grau
    x_regr = dados_ps['y1'].to_numpy().reshape(-1,1)
    y_regr = dados_ps['x1'].to_numpy()
    polinomio_3g = PolynomialFeatures(degree = 3)
    X_poli_3g = polinomio_3g.fit_transform(x_regr)
    mod_x_y = LinearRegression()
    mod_x_y.fit(X_poli_3g, y_regr)
    r2_x_y = r2_score(y_regr, mod_x_y.predict(X_poli_3g))
    #
    r2_modelos = [r2_H_x, r2_H_y, r2_y_x, r2_x_y]
    modelos = [mod_H_x, mod_H_y, mod_y_x, mod_x_y]
    #
    return (modelos, r2_modelos)


def f_uso_mod_loc_ps(x_var_ind, nome_var_dep, modelos):
    '''Função para usar os modelos locais estimados com: f_gera_mod_locais_ps
        Entradas:
        x_var_ind = valor da variável independente para o ual deseja calcular
                    a variável dependente
        nome_var_dep = string com o nome da variável dependente, podendo ser os
                    seguintes: 'Hl', 'Hv', 'y1' e 'x1'
        modelos = listas dos modelos estimados anteriormente
        Saidas:
        resp = valor calculado para a variável dependente
    '''
    #
    polinomio_2g = PolynomialFeatures(degree = 2)
    polinomio_3g = PolynomialFeatures(degree = 3)
    #
    if (nome_var_dep == 'Hl'):
        mod = modelos[0]
        resp = mod.predict(polinomio_2g.fit_transform(np.array([x_var_ind]).reshape(-1,1)))[0]
    elif (nome_var_dep == 'Hv'):
        mod = modelos[1]
        resp = mod.predict(polinomio_2g.fit_transform(np.array([x_var_ind]).reshape(-1,1)))[0]
    elif (nome_var_dep == 'y1'):
        mod = modelos[2]
        resp = mod.predict(polinomio_3g.fit_transform(np.array([x_var_ind]).reshape(-1,1)))[0]
    elif (nome_var_dep == 'x1'):
        mod = modelos[3]
        resp = mod.predict(polinomio_3g.fit_transform(np.array([x_var_ind]).reshape(-1,1)))[0]
    #
    return resp

def f_gera_dados_diag_H_ps(P_eq, npg, lista_componentes,dados):
    '''Função que gera o dataframe com os dados para a preparação do gráfico suporte
        do Método Ponchon-Savarit
        Entrada:
        P_eq = pressão de equilíbrio em mmHg
        npg = quantidade de pontos gerados
        lista_componentes = 
        dados = 
        Saida:
        dados_ps = dataframe com os dados para a construção do gráfico e dos modelos locais 
                para o Método Ponchon-Savarit
    '''
    dados_ps = f_gerar_dados_elv_2c_bd(P_eq, npg, lista_componentes,dados)
    DH_vap = np.zeros((npg,))
    Hig_v  = np.zeros((npg,))
    for i, row in dados_ps.iterrows():
        #print(i, row['T'])
        DH_vap_vet =  f_DHvap_Watson_db(row['T'], lista_componentes, dados)
        y_vez = np.array([row['y1'], (1.0-row['y1'])])
        DH_vap[i] = y_vez @ DH_vap_vet
        Hig_v[i] = f_H_vap_ig_stream(y_vez, 273.15, row['T'], lista_componentes, dados)[0]
    dados_ps['DH_vap_Watson'] = DH_vap
    dados_ps['Hig_v'] = Hig_v
    dados_ps['Hig_l'] = dados_ps['Hig_v'] - dados_ps['DH_vap_Watson']
    return dados_ps


def f_gera_diag_entalpia_ps(H_min_graf, H_max_graf, dados_ps):
    fig_ps, ax_ps = plt.subplots(num='PS_graf', figsize =(8,6))
    # pontos e linhas do gráfico
    ax_ps.plot(dados_ps['x1'], dados_ps['Hig_l'], 'b', label='liquido saturado')
    ax_ps.plot(dados_ps['y1'], dados_ps['Hig_v'], 'r', label='vapor saturado')
    #ax_ps.plot(T_BP_comp, DH_vap_comp*4.184, 'go', label='Exp')
    #ax_ps.plot(T_grafico, cp_modelo, 'k', label= 'Mod')
    # Limites dos eixos
    plt.ylim((H_min_graf, H_max_graf))
    # Adicionando texto nos eixos - descrição
    ax_ps.set_xlabel(r'fração molar $x_1$ ou $y_1$')
    ax_ps.set_ylabel(r'Entalpia [$J/mol$]')
    # Adicionando título para a figura
    ax_ps.set_title(r'Diagrama Entalpia x composição')
    # Adicionando um texto
    ax_ps.text(0.4, 13000.0, r'@$P_{eq} = 760.0 \, mmHg$')
    # Adicionando uma legenda
    ax_ps.legend(loc='upper left')
    # Adicionando linha vertical
    #ax_ps.vlines(z[0],H_min_graf, H_max_graf, colors='m', linestyles='dashed')
    #ax_ps.axvline(z[0], colors='k', linestyles=':')
    #ax_ps.axvline(z[0], colors='k', linestyles=':')
    #ax_ps.axvline(z[0], colors='k', linestyles=':')
    #ax1.axvline(sol_flash['y_eq'][0])
    # Adicionando linha horizontal
    #ax1.hlines(T_flash,0, 1, colors='y', linestyles='solid')
    # Adicionando grade
    ax_ps.grid()
    # Ajustando os ticks
    from matplotlib.ticker import (MultipleLocator) #AutoMinorLocator
    ax_ps.xaxis.set_major_locator(MultipleLocator(0.10))
    ax_ps.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax_ps.yaxis.set_major_locator(MultipleLocator(10000.0))
    ax_ps.yaxis.set_minor_locator(MultipleLocator(1000.0))
    ax_ps.tick_params(which='minor', length=4, color='r')
    ax_ps.grid(b=True, which='major', color='k', linestyle='-')
    ax_ps.grid(b=True, which='minor', color='lightgray', linestyle='--')
    plt.show()
    return (fig_ps, ax_ps)


def f_Pinch(z, fv, modelos_ps):
    '''Função que determina o ponto de pinch
        Solução do sistema dormado pela equação de flash e a curva de equilíbrio, ou ainda,
        ponto no qual a reta de flash intercepta a curva de equilíbrio
        Entradas:
        z = composição da carga
        fv = fração vaporizada da carga - condição da carga
        modelos_ps = modelos locais polinomiais para as curvas do diagrama de
                    fases e de entalpia x composição
        Saídas:
        dicionário com os valores de x_p e y_p = composição do ponto de pinch/ELV
    '''
    q = 1 - fv
    a_flash = -(q/(1 - q))
    b_flash = (1/(1 - q))*z[1]
    def f_sol(x, q, z, modelos_ps, a_flash, b_flash):
        residuo = a_flash*x + b_flash - f_uso_mod_loc_ps(x, 'y1', modelos_ps)
        return residuo
    # f_sol(0.3,q,z,mod_yeq, a_flash, b_flash)
    # x_eq <- uniroot(f_sol,interval = c(0,1), q,z,mod_yeq, a_flash, b_flash)$root
    x_guest = 0.5
    x_eq = fsolve(f_sol, x_guest, args=(q, z, modelos_ps, a_flash, b_flash))[0]
    y_eq = f_uso_mod_loc_ps(x_eq, 'y1', modelos_ps)
    return {'x_p': x_eq, 'y_p': y_eq}


def f_H_carga(z, fv, modelos_ps):
    ''' Função que calcula a entalpia composta da carga
        Obs.: somente para:  0 <= fv <= 1
    '''
    q = 1 - fv
    res_Pinch = f_Pinch(z, fv, modelos_ps)
    #HFL = f_HLx(res_Pinch$x_p, mod_HLx)
    HFL = f_uso_mod_loc_ps(res_Pinch['x_p'], 'Hl', modelos_ps)
    #HFV <- f_HVy(res_Pinch$y_p, mod_HVy)
    HFV = f_uso_mod_loc_ps(res_Pinch['y_p'], 'Hv', modelos_ps)
    HF  = (1 - q)*HFV + q*HFL
    return {'HF': HF, 'HFV': HFV, 'HFL': HFL}


def f_reta_2p(p1,p2):
    ''' Função que calcula os coeficientes da reta que passa pelos pontos p1 e p2
    '''
    # p1 <- c(y1,x1)
    # p2 <- c(y2,x2)
    y1 = p1[0]
    x1 = p1[1]
    y2 = p2[0]
    x2 = p2[1]
    #
    # coeficiente angular
    #
    a = (y1 - y2)/(x1 - x2)
    #
    # coeficiente linear
    #
    b = (y2*x1 - y1*x2)/(x1 - x2)
    #
    return {'b': b, 'a': a}


def f_mostra_linha(p1, p2, cor, ls):
    ''' Função que mostra no gráfico ativo uma linha ligando os pontos p1 e p2
        Entrada:
        p1 = coordenadas do ponto 1 (y1, x1) - tupla
        p2 = coordenadas do ponto 2 (y2, x2) - tupla
        cor = cor da linha
        ls  = line style

        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    '''
    x_v = np.array([p1[1], p2[1]])
    y_v = np.array([p1[0], p2[0]])
    #plt.figure(fig.number)
    plt.plot(x_v, y_v, color=cor, linestyle=ls, marker='o', markerfacecolor=cor, markersize=5.0)
    return




