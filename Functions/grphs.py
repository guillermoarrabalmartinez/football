def player_comparator(player1, player2):
    x1 = players[players["player_name"] == player1]
    x1 = x1.groupby(["player_name"])[cols].mean()

    x2 = players[players["player_name"] == player2]
    x2 = x2.groupby(["player_name"])[cols].mean()

    z = pd.concat([x1, x2]).transpose().reset_index()
    z = z.rename(columns={"index": "attributes", player1: player1, player2: player2})
    z.index = z.attributes
    z[[player1, player2]].plot(kind="barh",
                               figsize=(8, 12),
                               colors=["orange", "grey"],
                               linewidth=1,
                               width=.7,
                               edgecolor=["k"] * z["attributes"].nunique()
                               )
    plt.xlabel("mean value")
    plt.legend(loc="best", prop={"size": 15})
    plt.grid(True, alpha=.3)
    plt.title(player1 + "  vs  " + player2)