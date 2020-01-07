from deuces import Deck, Evaluator, Card
import numpy as np
from V0.simulation.feature_engineering.proba_estimation import estimate_proba

class Game:

    version = 0.0

    def __init__(self, n_players):
        self.deck = Deck()
        self.hands = [list(x) for x in zip(self.deck.draw(n_players),self.deck.draw(n_players))]
        self.board = []

        self.n_players = n_players

        self.bets = np.zeros(n_players) + 10
        self.money= np.ones(n_players)*90

        self.ins = np.ones(n_players).astype(bool)
        self.all_ins = np.zeros(n_players).astype(bool)

        self.round = 0

    def next_round(self, decision_algorithm):
        """Generating a round, using the decision algorithm to make the bets
            (Nothing to adapt here)"""

        # card increment based on which round we are in
        if self.round == 0:
            pass
        elif self.round == 1:
            self.board += self.deck.draw(3)
        elif self.round == 2:
            self.board += [self.deck.draw(1)]
        elif self.round == 3:
            self.board += [self.deck.draw(1)]
        elif self.round > 3:
            return

        # extracting features to feed to the decision algorithm

        features = self.__get_features()

        # applying the decision algorithm
        decisions = [decision_algorithm(f) if is_in else None
                     for f,is_in in zip(features, self.ins) ]

        # applying the decision (to adapt to the format of the decision output)
        self.__update_game(decisions)

        # if the game is at the end, we attribute the gains
        if self.round == 3:
            self.__finalize_game()

        # updating the round number
        self.round += 1



    def __update_game(self, decisions:list):
        """
        applying the decision from the algorithm to the game state
        (Nothing to adapt here)
        :param decisions: list, output from the decision algorithm
        :return:
        """

        # updating who folded
        self.__fold(decisions)
        # deciding on a raise value based on the decision ouptut
        raise_value = self.__raise_value(decisions)
        # checking who goes all in and updating the bets
        all_ins = (self.money <= raise_value) & self.ins
        self.bets[all_ins] += self.money[all_ins]
        self.money[all_ins] = 0.
        # removing all ins pplayer from the pool
        self.all_ins |= all_ins
        self.ins &= ~self.all_ins
        # updating the bets for the other players
        self.bets[self.ins] += raise_value
        self.money[self.ins] -= raise_value


    def __finalize_game(self):
        """
        Giving all the money to the player with the best hand
        (Nothing to adapt here)
        :return:
        """

        evaluator = Evaluator()
        hand_strength = np.array([evaluator.evaluate(hand, self.board) for hand in self.hands])

        if any(~(self.all_ins | self.ins)):
            hand_strength[~(self.all_ins | self.ins)] = np.nan

        if any(self.all_ins | self.ins):
            winner = np.nanargmin(hand_strength)
            # computing how much money the winner is getting from the others
            money_transfer = np.min([self.bets,[self.bets[winner]]*self.n_players],axis=0)
            self.money[winner] += np.sum(money_transfer)
            self.bets = self.bets - money_transfer

        # redistributing what hasn't been won to the players
        self.money += self.bets
        self.bets *= 0

    def __get_features(self):
        """
        Creating the features for the decision algorithm
        (to be adapted in development)
        :return:
        """

        feature_dict = {}

        feature_dict['player'] = list(np.arange(self.n_players))

        feature_dict['winning_proba'] = [estimate_proba(hand,
                                                        self.board,
                                                        self.n_players,
                                                        n_simul=100)
                                         for hand in self.hands]

        feature_dict['money_to_gain'] = [sum(self.bets)]*self.n_players

        for round in range(4):
            feature_dict[f'round {round}'] = [round == self.round]*self.n_players

        return feature_dict

    def __raise_value(self, decisions: list) -> float:
        """
        Implementation on how to decide the raise value based on the decision output
        (to be adapted in development)
        :param decisions:
        :return:
        """

        return np.max( np.min( [decisions, self.money], axis = 0 ))



    def __fold(self, decisions: list):
        """
        Implementation of how to decide if a player folds based on the decision output
        (to be adapted in development)
        :param decision: output of the decision algorithm
        """

        return np.array(decisions) < 0

    def display(self):
        for i in range(self.n_players):

            if self.ins[i]:
                status = "in"
            elif self.all_ins[i]:
                status = "all in"
            else:
                status = "out"

            print("player %d: \t Money: %d \t Bets: %d \t Status: %s "
                  %(i, self.money[i], self.bets[i],status))

            Card.print_pretty_cards(self.hands[i])

        print("board:")
        Card.print_pretty_cards(self.board)

