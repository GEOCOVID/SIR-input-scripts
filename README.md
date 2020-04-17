# SEIHUDR-input-scripts
Scripts and input data to feed GaGa Ninja's MATLAB code for SEIHUDR Model

Descrição:

**adjust_temporal_serie_4_br.py** -> Script que consome a mais atual série temporal dos estados de Wesley e o arquivo "estado_sumario". Gera o set de parâmetros para o modelo de EDOs com fitting de alguns e também condições iniciais. Gera também gráficos bonitinhos comparando a curva fittada com os pontos observados.

**generate_pop_municipalities.py** -> Script que consome a mais atual série temporal das cidades de Wesley e o arquivo "popBR_". Gera uma planilha com os valores das variáveis do Modelo preditos por ele para o dia em que o programa foi rodado, com base no começo da série temporal que cada cidade possui.
