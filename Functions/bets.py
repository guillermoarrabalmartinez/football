def filter_country_selected(w, matches):
    '''En función del valor seleccionado en el desplegable filtra por las cuotas de la casa elegida'''

    if len(w.value) > 0:
        if w.value[0] == 'Bet365':
            a = ['match_api_id', 'B365H', 'B365D', 'B365A'];
            c = w.value
        elif w.value[0] == 'Bet&Win':
            a = ['match_api_id', 'BWH', 'BWD', 'BWA'];
            c = w.value[0]
        elif w.value[0] == 'William Hill':
            a = ['match_api_id', 'WHH', 'WHD', 'WHA'];
            c = w.value[0]
        elif w.value[0] == 'VcBet':
            a = ['match_api_id', 'VCH', 'VCD', 'VCA'];
            c = w.value[0]
        elif w.value[0] == 'Pinnacle':
            a = ['match_api_id', 'PSH', 'PSD', 'PSA'];
            c = w.value[0]
        elif w.value[0] == 'Interwetten':
            a = ['match_api_id', 'IWH', 'IWD', 'IWA'];
            c = w.value[0]
        elif w.value[0] == 'Ladbrokes':
            a = ['match_api_id', 'LBH', 'LBD', 'LBA'];
            c = w.value[0]
        else:
            a = ['match_api_id', 'B365H', 'B365D', 'B365A']
            c = 'Bet365'
    else:
        a = ['match_api_id', 'B365H', 'B365D', 'B365A']
        c = 'Bet365'

    return a, c


def apply_model_to_bets(test, y_pred, clasificador, bets, columnas, casa_apuestas, matches, bet_amount):
    ''' aplicamos el modelo elegido a las apuestas elegidas con el importe introducido en bet_amount'''

    test['y_pred'] = y_pred
    test['probab_lose'] = clasificador.predict_proba(X_test)[:, 0]
    test['probab_draw'] = clasificador.predict_proba(X_test)[:, 1]
    test['probab_win'] = clasificador.predict_proba(X_test)[:, 2]

    test = test.merge(bets[columnas], right_on='match_api_id', left_on='match_api_id',
                      how='left').rename(
        columns={columnas[1]: 'CouteWinH', columnas[2]: 'CouteDrawH', columnas[3]: 'CouteLoseH'})
    test = test.merge(matches[['match_api_id', 'Country', 'League', 'Home_team_name', 'Away_team_name']],
                      right_on='match_api_id', left_on='match_api_id', how='left')
    ### En cada ejemplo apostaremos la cantidad indicada e.x. 100€ a cada partido
    bet_amount = bet_amount
    test['potential_outcome'] = np.where(test.y_pred == 1, test['CouteWinH'] * bet_amount, np.where(test.y_pred == 0,
                                                                                                    test[
                                                                                                        'CouteDrawH'] * bet_amount,
                                                                                                    test[
                                                                                                        'CouteLoseH'] * bet_amount))
    test['real_outcome'] = np.where(test.y_pred == test.target, test['potential_outcome'] - bet_amount,
                                    bet_amount * (-1))
    columns = ['lose', 'draw', 'win']
    strategies = [0.66, 0.74, 0.8]
    strat = []
    for i in strategies:
        name = 'real_outcome_strategy_' + str(i * 100) + "%"
        test[name] = np.where(test[['probab_' + p for p in columns]].max(axis=1) >= i, test['real_outcome'], 0)
    print(casa_apuestas, "\n")
    print("Confiamos en todas nuestras predicciones: €", test['real_outcome'].sum())
    print('Yield % ', round((test['real_outcome'].sum() / (test['real_outcome'].count() * bet_amount)) * 100, 2)
          , 'Total Apostado: ', test['real_outcome'].count() * bet_amount)
    print('\nEstrategia A, probabilidad superior al 66%: €', test['real_outcome_strategy_66.0%'].sum())
    print('Yield % ',
          round(test['real_outcome_strategy_66.0%'].sum() / (test.target[test['real_outcome_strategy_66.0%'] != 0
                                                                         ].count() * bet_amount) * 100, 2),
          'Total Apostado: ', test.target[test['real_outcome_strategy_66.0%'] != 0
                                          ].count() * bet_amount)
    print('\nEstrategia B, probabilidad superior al 74%: €', test['real_outcome_strategy_74.0%'].sum())
    print('Yield % ',
          round(test['real_outcome_strategy_74.0%'].sum() / (test.target[test['real_outcome_strategy_74.0%'] != 0
                                                                         ].count() * bet_amount) * 100, 2),
          'Total Apostado: ', test.target[test['real_outcome_strategy_74.0%'] != 0
                                          ].count() * bet_amount)
    print('\nEstrategia C, probabilidad superior al 80%: €', round(test['real_outcome_strategy_80.0%'].sum(), 2))
    print('Yield % ',
          round(test['real_outcome_strategy_80.0%'].sum() / (test.target[test['real_outcome_strategy_80.0%'] != 0
                                                                         ].count() * bet_amount) * 100, 2),
          'Total Apostado: ', test.target[test['real_outcome_strategy_80.0%'] != 0
                                          ].count() * bet_amount)
    test.to_excel(ruta + "testbet.xlsx", index=False)



    