
#Variable target y goal difference
def create_stats(matches):
    '''Vamos a crear el campo diferencia de goles y status del partido, victoria o derrota'''
    #Creamos campo diferencia de goles
    matches['Goal_Difference'] = matches.home_team_goal - matches.away_team_goal
    matches['home_status'] = np.where(matches['Goal_Difference'] > 0, 'W',
                             np.where(matches['Goal_Difference'] < 0, 'L', 'D'))
    matches['target'] = np.where(matches['home_status'] == 'W', 1,
                             np.where(matches['home_status'] == 'L', -1, 0))
    matches['homepos'] = matches['homepos'].fillna('50')
    matches['awaypos'] = matches['awaypos'].fillna('50')
    print(matches.shape)
    home_players = ["home_player_" + str(x) for x in range(1, 12)]
    away_players = ["away_player_" + str(x) for x in range(1, 12)]
    return matches , home_players, away_players

#Datos de Partidos
def get_fifa_stats(match, player_stats):
    ''' Aggregates fifa stats for a given match. '''

    # Define variables
    match_id = match.match_api_id
    date = match['date']
    player_stats_new = pd.DataFrame()
    names = []
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5", "home_player_6",
               "home_player_7", "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
               "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6", "away_player_7",
               "away_player_8", "away_player_9", "away_player_10", "away_player_11"]

    for player in players:
        # Sacamos id y estadisticas
        player_id = match[player]
        # Sacamos estadísticas mas actuales posibles
        stats = player_stats[player_stats.player_api_id == player_id]
        # Identify current stats
        current_stats = stats[stats.date < date].sort_values(by='date', ascending=False)[:1]
        if np.isnan(player_id) == True:
            overall_rating = pd.Series(0)
        else:
            current_stats.reset_index(inplace=True, drop=True)
            overall_rating = pd.Series(current_stats.loc[0, "overall_rating"])
        # Asignamos el nombre
        name = "overall_rating_{}".format(player)
        names.append(name)
        # Agregamos al dataframe que estamos creando
        player_stats_new = pd.concat([player_stats_new, overall_rating], axis=1)

    player_stats_new.columns = names
    player_stats_new['match_api_id'] = match_id

    player_stats_new.reset_index(inplace=True, drop=True)

    # Devolvemos estadísticas del jugador
    return player_stats_new.ix[0]



def create_fifa_data(matches, players, path = None ):
    ''' Iteramos por todos los partidos para sacar las estadísticas con la función anterior '''
    #Como la función tarda mucho podemos almacenar la información
    data_existe = False
    if data_existe == True:
        fifa_data = pd.read_pickle(path)
    else:
        start = time()
        #Apicamos la función anterior a todos y cada uno de los partidos get_fifa_stats for each match
        datos_fifa = matches.apply(lambda x :get_fifa_stats(x, players), axis = 1)
        end = time()
        print("Datos recogidos en{:.1f} minutos".format((end - start)/60))
    datos_fifa.replace(0, np.nan, inplace=True)
    return datos_fifa


#Fill na
def fill_na_fifa_stats(matches, home_players, away_players):
    columns =  ['league_id', 'season','home_team_api_id']
    columns2 =  ['league_id', 'season']
    for jugador in home_players:
        matches['overall_rating_' + jugador] = matches.groupby(columns)['overall_rating_' + jugador
                                                                       ].apply(lambda x: x.fillna(x.mean()))
        matches['overall_rating_' + jugador] = matches.groupby(columns2)['overall_rating_' + jugador
                                                                       ].apply(lambda x: x.fillna(x.mean()))
        matches['overall_rating_' + jugador] = matches['overall_rating_' + jugador].fillna(50)
    for jugador in away_players:
        matches['overall_rating_' + jugador] = matches.groupby(columns)['overall_rating_' + jugador
                                                                       ].apply(lambda x: x.fillna(x.mean()))
        matches['overall_rating_' + jugador] = matches.groupby(columns2)['overall_rating_' + jugador
                                                                       ].apply(lambda x: x.fillna(x.mean()))
        matches['overall_rating_' + jugador] = matches['overall_rating_' + jugador].fillna(50)
    print("Actualizadas Estadísticas Jugadores con NA")
    return matches

def create_fifa_overall_stats(matches, home_players, away_players):
    #Diferencias rating home y away
    matches['overall_rating_home'] = matches[['overall_rating_' + p for p in home_players]].sum(axis=1)
    matches['overall_rating_away'] = matches[['overall_rating_' + p for p in away_players]].sum(axis=1)
    matches['overall_rating_difference'] = matches['overall_rating_home'] - matches['overall_rating_away']
    #Characteristics
    matches['min_overall_rating_home'] = matches[['overall_rating_' + p for p in home_players]].min(axis=1)
    matches['min_overall_rating_away'] = matches[['overall_rating_' + p for p in away_players]].min(axis=1)

    matches['max_overall_rating_home'] = matches[['overall_rating_' + p for p in home_players]].max(axis=1)
    matches['max_overall_rating_away'] = matches[['overall_rating_' + p for p in away_players]].max(axis=1)

    matches['mean_overall_rating_home'] = matches[['overall_rating_' + p for p in home_players]].mean(axis=1)
    matches['mean_overall_rating_away'] = matches[['overall_rating_' + p for p in away_players]].mean(axis=1)

    matches['std_overall_rating_home'] = matches[['overall_rating_' + p for p in home_players]].std(axis=1)
    matches['std_overall_rating_away'] = matches[['overall_rating_' + p for p in away_players]].std(axis=1)

    matches = matches[matches['overall_rating_home'] >= 1]
    matches = matches[matches['overall_rating_away'] >= 1]
    print("Overall fifa stats created")
    return matches


def ultimos_partidos(matches, dates, equipo, z=8):
    '''queremos analizar ultimos z partidos de un equipo'''
    team_match = matches[(matches['home_team_api_id'] == equipo) | (matches['away_team_api_id'] == equipo)]
    last_match = team_match[team_match.date < dates].sort_values(by='date', ascending=False).iloc[0:z, :]
    return last_match


def goles_anotados_y_recibidos(matches, equipo):
    ''' cuantos goles han marcado en casa y fuera'''
    goles_recibidos_casa = int(matches.home_team_goal[matches.away_team_api_id == equipo].sum())
    goles_recibidos_fuera = int(matches.away_team_goal[matches.home_team_api_id == equipo].sum())

    goles_casa = int(matches.home_team_goal[matches.home_team_api_id == equipo].sum())
    goles_fuera = int(matches.away_team_goal[matches.away_team_api_id == equipo].sum())

    total = goles_fuera + goles_casa
    total_recibidos = goles_recibidos_fuera + goles_recibidos_casa

    return total, total_recibidos


def consecutive_wins(matches, equipo):
    '''sacaremos victorias'''
    home_win = int(matches.home_team_goal[(matches.home_team_api_id == equipo) & (
                matches.home_team_goal > matches.away_team_goal)].count())
    away_win = int(matches.away_team_goal[(matches.away_team_api_id == equipo) & (
                matches.away_team_goal > matches.home_team_goal)].count())

    wins = away_win + home_win
    return wins


def consecutive_draws(matches, equipo):
    '''sacaremos empates'''
    home_draw = int(matches.home_team_goal[(matches.home_team_api_id == equipo) & (
                matches.home_team_goal == matches.away_team_goal)].count())
    away_draw = int(matches.away_team_goal[(matches.away_team_api_id == equipo) & (
                matches.away_team_goal == matches.home_team_goal)].count())

    draws = home_draw + away_draw
    return draws


def last_games_against(df, dates, home_team, away_team, z=6):
    '''ultimos enfrentamientos entre equipos'''
    home_match = df[(df['home_team_api_id'] == home_team) & (df['away_team_api_id'] == away_team)]
    away_match = df[(df['home_team_api_id'] == away_team) & (df['away_team_api_id'] == home_team)]
    total_match = pd.concat([away_match, home_match])
    try:
        last_matches = total_match[total_match.date < dates].sort_values(by='date', ascending=False).iloc[0:z, :]
    except:
        last_matches = total_match[total_match.date < dates].sort_values(by='date', ascending=False
                                                                         ).iloc[0:total_match.shape[0], :]
        if (last_matches.shape[0] > x): print("Mistake obtaining matches")
    return last_matches


def possesion_statistics(matches, equipo):
    '''posesion como local y visitante'''
    home_possesion = int(matches.homepos[matches.home_team_api_id == equipo].sum())
    away_possesion = int(matches.awaypos[matches.away_team_api_id == equipo].sum())

    total_possesion = home_possesion + away_possesion
    return total_possesion


def create_team_stats(partido, matches, z=8):
    '''creamos las estadísticas para cada partido'''

    home_team = partido['home_team_api_id']
    away_team = partido['away_team_api_id']
    dates = partido['date']
    # Ultimos partidos en casa y fuera
    home_team_matches = ultimos_partidos(matches, dates, home_team, z=10)
    away_team_matches = ultimos_partidos(matches, dates, away_team, z=10)
    # Ultimos partidos entre los dos equipos:
    matches_against = last_games_against(matches, dates, home_team, away_team, z=4)
    # Goles recibidos en casa y fuera
    home_goals, home_goals_recieved = goles_anotados_y_recibidos(home_team_matches, home_team)
    away_goals, away_goals_recieved = goles_anotados_y_recibidos(away_team_matches, away_team)
    # Posesiones en casa y fuera
    home_possesion = possesion_statistics(home_team_matches, home_team)
    away_possesion = possesion_statistics(away_team_matches, away_team)

    # Vamos a crear nuevo df
    stats = pd.DataFrame()
    stats.loc[0, 'match_api_id'] = int(partido['match_api_id'])
    stats.loc[0, 'league_id'] = int(partido['league_id'])
    stats.loc[0, 'home_team_goals_scored_lastmatches'] = home_goals
    stats.loc[0, 'away_team_goals_scored_lastmatches'] = away_goals
    stats.loc[0, 'home_team_goals_recieved_lastmatches'] = home_goals_recieved
    stats.loc[0, 'away_team_goals_recieved_lastmatches'] = away_goals_recieved
    stats.loc[0, 'home_possesion'] = home_possesion
    stats.loc[0, 'away_possesion'] = away_possesion
    stats.loc[0, 'home_team_dif_goals'] = home_goals - home_goals_recieved
    stats.loc[0, 'away_team_dif_goals'] = away_goals - away_goals_recieved
    stats.loc[0, 'games_won_h_team'] = consecutive_wins(home_team_matches, home_team)
    stats.loc[0, 'games_won_a_team'] = consecutive_wins(away_team_matches, away_team)
    stats.loc[0, 'games_against_won'] = consecutive_wins(matches_against, home_team)
    stats.loc[0, 'games_against_lost'] = consecutive_wins(matches_against, away_team)
    stats.loc[0, 'games_against_draw'] = consecutive_draws(matches_against, away_team)
    stats.loc[0, 'games_draw_h_team'] = consecutive_draws(home_team_matches, home_team)
    stats.loc[0, 'games_draw_a_team'] = consecutive_draws(away_team_matches, away_team)

    return stats.loc[0]