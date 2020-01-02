from deuces import Deck, Evaluator
import numpy as np
from feature_engineering.proba_estimation import estimate_proba


class Game:

    version = 0.0

    def __init__(self, n_players):
        self.deck = Deck()
        self.hands = [list(x) for x in zip(self.deck.draw(n_players),self.deck.draw(n_players))]
        self.board = []

        self.n_players = n_players

        self.bets = np.zeros(n_players)
        self.money= np.ones(n_players)*100

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
            self.board += self.deck.draw(1)
        elif self.round == 3:
            self.board += self.deck.draw(1)
        elif self.round > 3:
            return

        # extracting features to feed to the decision algorithm
        features = self.__get_features()

        # applying the decision algorithm
        decisions = [decision_algorithm(f) for f in features]

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
        self.__fold(decisions)

        raise_value = self.__raise_value(decisions)

        all_ins = (self.money >= raise_value) & self.ins
        self.bets[all_ins] += self.money[all_ins]
        self.money[all_ins] = 0.

        self.all_ins |= all_ins
        self.ins &=  ~self.all_ins

        self.bets[self.ins] += raise_value
        self.money[self.ins] -= raise_value


    def __finalize_game(self):
        """
        Giving all the money to the player with the best hand
        (Nothing to adapt here)
        :return:
        """
        evaluator = Evaluator()
        winner = np.argmax( [evaluator.evaluate(hand, self.board) for hand in self.hands] )
        self.money[winner] = sum(self.bets)


    def __get_features(self):
        """
        Creating the features for the decision algorithm
        (to be adapted in development)
        :return:
        """

        feature_dict = {}
        feature_dict['winning_proba'] = [estimate_proba(hand,
                                                        self.board,
                                                        self.n_players,
                                                        n_simul=100)
                                         for hand in self.hands]

        feature_dict['money_to_gain'] = [sum(self.bets)]*self.n_players

        for round in range(4):
            feature_dict[f'round {round}'] = [round == self.round]*self.n_players

        return feature_dict

    def __raise_value(self, decisions: list)->float:
        """
        Implementation on how to decide the rais value based on the decision output
        (to be adapted in development)
        :param decisions:
        :return:
        """

    def __fold(self, decisions:list):
        """
        Implementation of how to decide if a player folds based on the decision output
        (to be adapted in development)
        :param decision: output of the decision algorithm
        """


