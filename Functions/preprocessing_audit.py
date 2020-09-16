def print_datatype(datatype, df):
    """ Parámetro data type ('object', 'int64', 'float64')
        Parámetro df: nombre del dataframe
        Devuelve Dataframe con nombre de columna, valores distintos, cantidad de valores distinos y número de NA."""
    list = df.dtypes[df.dtypes == datatype].index.tolist()
    x, y, z = [], [], []
    for i in list:
        a = df[i].unique()
        x.append(a)
        y.append(len(a))
        z.append(df[i].isnull().sum())
    tempdf = pd.DataFrame({'Unique Values': x, 'qty': y, 'na': z}, index=[list])
    print("\nData Frame 'Matches' contains following columns of", datatype, "data")
    return tempdf.sort_values('qty')

def label(data):
    if data["home_team_goal"] > data["away_team_goal"]:
        return data["Home_team_name"]
    elif data["away_team_goal"] > data["home_team_goal"]:
        return data["Away_team_name"]
    elif data["home_team_goal"] == data["away_team_goal"]:
        return "EMPATE"
