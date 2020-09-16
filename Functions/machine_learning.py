# Jornada desde la que queremos hacer predicciones:
def train_test_split_season_stage(match, season, n_stages):
    '''Esta funcion hace el split de conjuntos, es mi train_test_split de Sklearn casero'''

    test = match[(match['season'] == '2015/2016') & (match['stage'] > n_stages) & (match['stage'] < 36)]
    train = match[~match['match_api_id'].isin(test.match_api_id.unique())]
    X_test = test.drop(columns={'date', 'season', 'stage', 'target', 'match_api_id'})
    y_test = test['target']
    X_train = train.drop(columns={'date', 'season', 'stage', 'target', 'match_api_id'})
    y_train = train['target']
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test, test, train


def filter_season_Country(matches, country):
    '''Filtra los conjuntos por el país indicado. Si se quieren dejar temporadas fuera modificar codigo comentado'''

    temporadas = pd.Series(matches['season'].unique())
    excluimos_temporadas = np.array([])  # ,'2008/2009', '2009/2010', '2010/2011', '2011/2012'])#, '2012/2013'])
    selected_seasons = temporadas[~temporadas.isin(excluimos_temporadas)]
    # Filtramos
    match = matches[matches['season'].isin(np.array(selected_seasons)) & (matches['Country'] == country)]
    match = match[match_labels].reset_index(drop=True)
    return match, country


def normalizar_conjuntos(X_train, X_test):
    '''normalizamos los conjuntos de datos para pasar por los algoritmos'''

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    return X_train, X_test


def regresion_logistica(X_train, y_train, X_test, y_test):
    '''clasificador de regresión logística'''

    clasificador = LogisticRegression()
    clasificador.fit(X_train, y_train)
    y_pred = clasificador.predict(X_test)
    y_pred_p = clasificador.predict_proba(X_test)

    predicciones = pd.DataFrame({'real': y_test, 'pred': y_pred,
                                 'prob_-1': y_pred_p[:, 0], 'prob_0': y_pred_p[:, 1], 'prob_1': y_pred_p[:, 2]})
    predicciones['hit'] = np.where((predicciones['real'] == predicciones['pred']), 1, 0)
    print('Accuracy en el modelo de regresión: ', sum(predicciones.hit) / len(predicciones))
    return y_pred, predicciones


# RANDOM FOREST
def Random_Forest(X_train, y_train, X_test, y_test, n_estimators, bootstrap, max_depth, max_features, min_samples_leaf,
                  min_samples_split):
    '''clasificador Random Forest'''

    clasificador = RandomForestClassifier(n_estimators=n_estimators, bootstrap=bootstrap, max_depth=max_depth,
                                          max_features=max_features, min_samples_leaf=min_samples_leaf,
                                          min_samples_split=min_samples_split)
    clasificador.fit(X_train, y_train)
    y_pred = clasificador.predict(X_test)
    return y_pred, clasificador


# SVClassifier
def Support_Vector_Classification(X_train, y_train, X_test, y_test, kernel, C_param, gamma):
    '''clasificador SVC'''

    clasificador = SVC(kernel=kernel, C=C_param, gamma=gamma, probability=True)
    clasificador.fit(X_train, y_train)
    y_pred = clasificador.predict(X_test)
    return y_pred, clasificador


# XgBoost
def XGBoost_Clasificator(X_train, y_train, X_test, y_test, learning_rate, n_estimators):
    '''clasificdor Xgboost'''

    clasificador = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
    clasificador.fit(X_train, y_train)
    y_pred = clasificador.predict(X_test)
    return y_pred, clasificador


# AdaBoost
def AdaBoost_Clasificator(X_train, y_train, X_test, y_test, learning_rate, n_estimators):
    '''Clasificador AdaBoost'''

    clasificador = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
    clasificador.fit(X_train, y_train)
    y_pred = clasificador.predict(X_test)
    return y_pred, clasificador


## KNeighbors Model
def Kneighbors_Clasificator(X_train, y_train, X_test, y_test, n_splits):
    '''clasificador Kneighbors'''

    seed = 42
    kfold = model_selection.KFold(n_splits=n_splits, random_state=seed)
    clasificador = KNeighborsClassifier()
    clasificador.fit(X_train, y_train)
    y_pred = clasificador.predict(X_test)
    return y_pred, clasificador


## Gaussian Model
def Gaussian_Clasificator(X_train, y_train, X_test, y_test, n_splits):
    '''Clasificador Gausiano Naive Bayes'''

    seed = 42
    kfold = model_selection.KFold(n_splits=n_splits, random_state=seed)
    clasificador = GaussianNB()
    clasificador.fit(X_train, y_train)
    y_pred = clasificador.predict(X_test)
    return y_pred


def Neuronal_Net_Classificator(X_train, y_train, X_test, y_test, batch_size, epochs, result):
    ''' función para ejecutar la red neuronal'''

    parada = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
    batch_size = batch_size
    epochs = epochs
    steps_per_epoch = len(X_train) / batch_size
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_dim=26, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='sigmoid')])

    optimizer = tf.keras.optimizers.Adagrad(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        steps_per_epoch=steps_per_epoch, validation_data=(X_test, y_test), callbacks=[parada])

    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print('\nAccuracy: %.2f' % (test_accuracy * 100))

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # Set the vertical range
    plt.show()

    # Matriz de confusion,

    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=1)
    if not 1 in y_pred:
        y_pred[-1] = 1
    y_pred = [-1 if x == 0 else 0 if x == 1 else 1 for x in y_pred]
    y_pred = pd.get_dummies(y_pred)
    n = len(result) + 1
    try:
        matriz_confusion = multilabel_confusion_matrix(y_test, y_pred)
        print(matriz_confusion)
        result.loc[n] = ["Neuronal Net"] + [test_accuracy] + [matriz_confusion[0][1][1]] + [
            matriz_confusion[1][1][1]] + [matriz_confusion[2][1][1]]
    except:
        result.loc[n] = ["Neuronal Net"] + [test_accuracy] + [0] + [0] + [0]
    return y_pred, result


def resultados_modelo(y_test, y_pred, result, model_name):
    ''' función que devuelve resultados del modelo y los incluye en el dataframe de resultados'''

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    TP = np.diag(confusion_matrix(y_test, y_pred))
    n = len(result) + 1
    result.loc[n] = [model_name] + [metrics.accuracy_score(y_test, y_pred)] + [TP[0]] + [TP[1]] + [TP[2]]
    return TP


def models_for_each_country(X_train, X_test, y_train, y_test):
    '''Ejecutamos cada uno de los modelos para el conjunto de datos cargado. Creamos DF con los resultados'''

    result = pd.DataFrame(
        columns=('Algoritmo_Utilizado', 'Accuracy', 'Aciertos_H.Lose', 'Aciertos_H.Draw', 'Aciertos_H.Victory'))
    y_pred, predicciones = regresion_logistica(X_train, y_train, X_test, y_test)
    print("\nLogistic Regression")
    TP = resultados_modelo(y_test, y_pred, result, "Logistic Regression")
    y_pred, clasificador = Random_Forest(X_train, y_train, X_test, y_test, 200, True, 80, 2, 5, 10)
    print("\nRandom Forrest")
    TP = resultados_modelo(y_test, y_pred, result, "Random Forrest")
    y_pred, clasificador = Support_Vector_Classification(X_train, y_train, X_test, y_test, 'linear', 1.0, 0.01)
    print("\nSupport Vector Classification")
    TP = resultados_modelo(y_test, y_pred, result, "Support Vector Classification")
    y_pred, clasificador = XGBoost_Clasificator(X_train, y_train, X_test, y_test, 0.01, 600)
    print("\nXGB Classifier")
    TP = resultados_modelo(y_test, y_pred, result, "XGB Classifier")
    y_pred, clasificador = AdaBoost_Clasificator(X_train, y_train, X_test, y_test, 0.01, 500)
    print("\nAdaBoost Classifier")
    TP = resultados_modelo(y_test, y_pred, result, "AdaBoost Classifier")
    y_pred, clasificador = Kneighbors_Clasificator(X_train, y_train, X_test, y_test, 2)
    print("\nK Neighbors Model")
    TP = resultados_modelo(y_test, y_pred, result, "K Neighbors Model")
    y_pred = Gaussian_Clasificator(X_train, y_train, X_test, y_test, 2)
    print("\nGaussian Model")
    TP = resultados_modelo(y_test, y_pred, result, "Gaussian Model")
    print("\nNeuronal Net")
    y_pred, result = Neuronal_Net_Classificator(X_train, y_train, X_test, y_test, 10, 150, result)

    return result


def Confusion_Matrix(y_test, y_pred):
    '''Representa gráficamente la matriz de confusión'''

    cm = confusion_matrix(y_test, y_pred)
    index = ['Away_Victory', 'Draw', 'Home_Victory']
    columns = ['Away_Victory', 'Draw', 'Home_Victory']
    cm_df = pd.DataFrame(cm, columns, index)
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(cm_df, annot=True, cmap="Greens")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('Valor Pronosticado', fontsize=15)  # x-axis label with fontsize 15
    plt.ylabel('Valor Real', fontsize=15)  # y-axis label with fontsize 15
    plt.show();