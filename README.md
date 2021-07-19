# Reinforcement Learning through learned model

Neste trabalho quero utilizar Word Models com Deep RL. Inspiração Dreamer V2 e World Models.

O objetivo deste sistema é a criação do modelo do ambiente e realizar a previsão dos passos subsequentes aos passos inicias. Ele utiliza o estado latente proveniente de um AutoEncoder Variacional (VAE) para a redução da dimensionalidade da observação e uma RNN como memória para a previsão dos passos futuros. É possível se tomar ações neste estado latente fazendo com que se possa treinar um agente nele.

Primeiramente pretendo utilizar um ambiente mais simples para desenvolver a técnica e entender suas minucias. Após o desenvolvimento inicial desejo aplicar este sistema em algum problema de robótica, que pode ser algo com carros autonômos, drones ou manipulação.

Abaixo tenho uma prévia das etapas iniciais a serem realizadas.

| Etapa |      Tarefa       |        Realização        |
| :---: | :---------------: | :----------------------: |
|   1   |       Ideia       |    :heavy_check_mark:    |
|   2   |        VAE        |    :heavy_check_mark:    |
|   3   |        RNN        |    :heavy_check_mark:    |
|   4   |      Deep RL      | :heavy_exclamation_mark: |
|   5   | Integrar sistemas | :heavy_exclamation_mark: |
|   6   |      Treinar      | :heavy_exclamation_mark: |



## VAE

|          Tarefa           |     Realização     |
| :-----------------------: | :----------------: |
|          Dataset          | :heavy_check_mark: |
|          Modelo           | :heavy_check_mark: |
|      Loop de treino       | :heavy_check_mark: |
| Salvar e carregar modelos | :heavy_check_mark: |



## LSTM

|          Tarefa           |     Realização     |
| :-----------------------: | :----------------: |
|          Modelo           | :heavy_check_mark: |
|      Loop de treino       | :heavy_check_mark: |
| Salvar e carregar modelos | :heavy_check_mark: |
|                           |                    |



## Deep-RL

| Tarefa | Realização |
| :----: | :--------: |
|        |            |
|        |            |
|        |            |



## Integrar sistemas

|           Tarefa           |        Realização        |
| :------------------------: | :----------------------: |
| Arquivo de hiperparâmetros | :heavy_exclamation_mark: |
|                            |                          |
|                            |                          |





# Relatório

1.-Introdução (objetivo geral, objetivo especifico, justificativa)

2.-Referencial teorico (VAE, Controlador, RL, DRL)

3.-Materiais e métodos (openai, gym, pytorch)

4.-Resultados(Loss VAE, REsultado do controlador sem memoria)

5.-Conclusões

