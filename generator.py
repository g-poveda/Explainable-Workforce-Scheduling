
import random
import math
import pandas as pd

class DisruptionScenario:

    FEW_DISRUPTIONS = "few_disruptions"
    LONG_DISRUPTIONS = "long_disruptions"
    MANY_DISRUPTIONS = "many_disruptions"
    MANY_TEAMS_AFFECTED = "many_teams_affected"
    ALL_TEAMS_AFFECTED = "all_teams_affected"
    ONE_TEAM_DROPPED = "one_team_dropped"
    TWO_TEAMS_DROPPED = "two_teams_dropped"
    DELAY = "delay"


def generate_from_scenario(category, horizon, teams, seed=0):

    if category == DisruptionScenario.FEW_DISRUPTIONS:
        params = dict(
            num_disruptions=list(range(1,3)),
            num_teams_affected = (1, math.ceil(len(teams)/2)),
            duration = [15,30,60,120]
        )

    elif category == DisruptionScenario.LONG_DISRUPTIONS:
        params = dict(
            num_disruptions=list(range(1, 4)),
            num_teams_affected = (1,1),
            duration = (120, 240, 360)
        )

    elif category == DisruptionScenario.MANY_DISRUPTIONS:
        params = dict(
            num_disruptions=list(range(5,11)),
            num_teams_affected = (1, math.ceil(len(teams))),
            duration = (15,30,60,120)
        )

    elif category == DisruptionScenario.MANY_TEAMS_AFFECTED:
        params = dict(
            num_disruptions=(1,1),
            num_teams_affected = (math.ceil(len(teams) / 2), len(teams)),
            duration = (15,30,60,120)
        )

    elif category == DisruptionScenario.ALL_TEAMS_AFFECTED:
        params = dict(
            num_disruptions=(1,1),
            num_teams_affected = len(teams),
            duration = (15,30,60,120)
        )

    elif category == DisruptionScenario.ONE_TEAM_DROPPED:
        params = dict(
            num_disruptions=list(range(1, 4)),
            num_teams_affected = (1,1),
            duration = horizon
        )

    elif category == DisruptionScenario.TWO_TEAMS_DROPPED:
        params = dict(
            num_disruptions=range(1, 4),
            num_teams_affected = (2,2),
            duration = horizon
        )

    # TODO: add delay scenario

    else:
        raise ValueError(f"Invalid disruption category: {category}")
    
    return generate_disruption(**params,horizon=horizon,teams=teams,seed=seed)


def generate_disruption(num_disruptions, num_teams_affected, duration, horizon, teams, seed=0):

    random.seed(seed)

    disruptions = []
    for _ in range(random.randint(*num_disruptions)):
        dur = random.choice(duration)
        start = random.randint(0, horizon - dur)
        end = start + dur

        teams_affected = random.sample(teams, random.randint(*num_teams_affected))

        for team in teams_affected:
            disruptions.append(dict(
                team_id = team,
                start_unavailable = start,
                end_unavailable = end
            ))

    return pd.DataFrame(disruptions)




    