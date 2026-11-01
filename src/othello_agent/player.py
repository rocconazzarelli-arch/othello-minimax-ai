import math
import random
import numpy as np

# If this file is inside a package, keep this:
from . import weights
# If you are testing as a single file, comment the line above and use:
# import tabelas


class JogadorFortissimo:
    """
    Very strong AI player based on positional evaluation + minimax + alpha-beta.
    It respects the same interface as PlayerAleatorio.
    """

    def __init__(self):
        # Board / weights info from tabelas.py
        self._idxcoef = weights.indice_coeficientes
        self._coords_coef = weights.coordenadas_com_coeficiente
        self._tabela_1 = weights.tabela_primeiro_jogador
        self._tabela_2 = weights.tabela_segundo_jogador

        self._lins = len(self._idxcoef)
        self._cols = len(self._idxcoef[0])

        # These will be set in nova_partida
        self._jogador = None         # +1 or -1 (or whatever the environment uses)
        self._tabela = None          # active weight table for this player

        # Search parameters – tune if you hit time problems
        self._max_depth_midgame = 4  # depth when there are still many empty squares
        self._max_depth_endgame = 6  # deeper search in late game (fewer empty squares)

    # ======================================================================
    # REQUIRED INTERFACE METHODS
    # ======================================================================

    def nova_partida(self, jogo, jogador, id_oponente=None):
        """
        Called at the start of each game.

        :param jogo:    Initial game state.
        :param jogador: This player's ID (1st or 2nd player).
        :param id_oponente: Optional opponent ID (unused here).
        """
        self._jogador = jogador
        # Choose corresponding weight table depending on whether we are
        # first or second player.
        self._tabela = self._tabela_1 if jogador == 1 else self._tabela_2

    def escolhe_jogada(self, jogo):
        """
        Main decision method. Chooses the move using minimax + alpha-beta.
        """
        jogadas = jogo.jogadas_legais()

        # If only one legal move, don't overthink it
        if len(jogadas) == 1:
            return jogadas[0]

        # Decide search depth based on how many empty squares remain
        empty = self._conta_vazios(jogo)
        if empty <= 10:
            max_depth = self._max_depth_endgame
        else:
            max_depth = self._max_depth_midgame

        # ROOT: we are the maximizing player
        melhor_valor = -math.inf
        melhor_jogada = random.choice(jogadas)  # fallback in case of tie

        # Simple move ordering at root: evaluate 1-ply to sort moves
        jogadas_ordenadas = sorted(
            jogadas,
            key=lambda j: self._avalia_jogada(grafo=jogo, jogada=j),
            reverse=True
        )

        for jogada in jogadas_ordenadas:
            novo_jogo = jogo.joga(jogada)
            valor = self._minimax(
                jogo=novo_jogo,
                profundidade=max_depth - 1,
                alpha=-math.inf,
                beta=math.inf,
                maximizando=False  # after we move, opponent moves
            )
            if valor > melhor_valor:
                melhor_valor = valor
                melhor_jogada = jogada

        return melhor_jogada

    def informa_propria_jogada(self, tabuleiro_antes, jogada, tabuleiro_depois):
        """
        We ignore this info for now (no learning online).
        You could log or implement online learning here if allowed.
        """
        pass

    def informa_jogada_oponente(self, tabuleiro_antes, jogada, tabuleiro_depois):
        """
        We ignore this info for now (no opponent modelling).
        """
        pass

    def informa_fim(self, jogo_final):
        """
        Called at the end of the game. Could be used to update statistics or
        learning. We keep it empty to stay deterministic.
        """
        pass

    # ======================================================================
    # SEARCH: MINIMAX + ALPHA-BETA
    # ======================================================================

    def _minimax(self, jogo, profundidade, alpha, beta, maximizando):
        """
        Standard minimax with alpha-beta pruning.
        We alternate between maximizing (our turn) and minimizing (opponent).
        """
        jogadas = jogo.jogadas_legais()

        # Terminal node or depth limit
        if profundidade == 0 or not jogadas:
            return self._avalia_posicao(jogo)

        if maximizando:
            valor = -math.inf
            # Optional: simple ordering by static eval
            for jogada in jogadas:
                novo_jogo = jogo.joga(jogada)
                filho = self._minimax(novo_jogo, profundidade - 1, alpha, beta, False)
                valor = max(valor, filho)
                alpha = max(alpha, valor)
                if beta <= alpha:
                    break  # beta cut-off
            return valor
        else:
            valor = math.inf
            for jogada in jogadas:
                novo_jogo = jogo.joga(jogada)
                filho = self._minimax(novo_jogo, profundidade - 1, alpha, beta, True)
                valor = min(valor, filho)
                beta = min(beta, valor)
                if beta <= alpha:
                    break  # alpha cut-off
            return valor

    # ======================================================================
    # EVALUATION
    # ======================================================================

    def _conta_vazios(self, jogo):
        """
        Counts empty cells on the board, assuming jogo.posicao((i,j)) == 0
        means empty.
        """
        vazios = 0
        for i in range(self._lins):
            for j in range(self._cols):
                if jogo.posicao((i, j)) == 0:
                    vazios += 1
        return vazios

    def _valor_pesos(self, jogo):
        """
        Linear combination of board values and learned weights.
        This is the core heuristic.
        """
        val = 0.0
        for i in range(self._lins):
            for j in range(self._cols):
                p = jogo.posicao((i, j))
                if p != 0:
                    # indice_coeficientes[i][j] gives index into the weight table
                    idx = self._idxcoef[i][j]
                    val += p * self._tabela[idx]

        # Perspective: if we are player -1, invert
        # (Assuming pieces of player 1 are +1, player 2 are -1)
        if self._jogador == -1:
            val = -val

        return val

    def _avalia_posicao(self, jogo):
        """
        Smooth bounded evaluation using tanh.
        Result in [-1, 1], where >0 is good for us, <0 is bad.
        """
        bruto = self._valor_pesos(jogo)
        # scaling not strictly necessary, but helps avoid saturation if weights are huge
        return float(np.tanh(bruto / 100.0))

    def _avalia_jogada(self, grafo, jogada):
        """
        Quick static evaluation of a move (1-ply), used for move ordering.
        """
        try:
            novo = grafo.joga(jogada)
        except Exception:
            # Illegal or problematic move: treat as very bad
            return -1e9
        return self._avalia_posicao(novo)


# ======================================================================
# FACTORY FUNCTION REQUIRED BY THE ASSIGNMENT
# ======================================================================

def cria_jogador():
    """
    Function required by the automatic evaluation script.
    Returns an object exposing the PlayerAleatorio interface.
    """
    return JogadorFortissimo()


# Backward-compatible alias retained for older evaluation scripts.
crea_player = cria_jogador
