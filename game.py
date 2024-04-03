import random
from IPython.display import clear_output

COLORS = ['blue', 'green', 'yellow', 'white', 'red']

class Tile():

    def __init__(self, color, number):

        self.color = color
        self.number = number


    def __str__(self):
        return self.color + str(self.number)

    def __repr__(self):
        return self.color + str(self.number)

    def encode(self):
        out = [-1, -1]
        if self.color != 'unk':
            out[0] = COLORS.index(self.color)
        if self.number != 'unk':
            out[1] = self.number

        return out
    
    
class Player():

    def __init__(self, game):
        self.tiles = []
        self.knowledge = []
        self.game = game

    def take(self):
        if len(self.game.board.deck) > 0:
            self.tiles.append(self.game.board.deck[-1])
            self.knowledge.append(Tile('unk', 'unk'))
            self.game.board.deck.pop(-1)


    def place(self, tile):

        if tile >= len(self.tiles):
          tile = len(self.tiles) - 1

        t = self.tiles[tile]

        if t.number == self.game.board.table[t.color] + 1:

            try:
                self.game.turn_reward += 10
            except:
                pass

            self.game.board.table[t.color] += 1
            self.tiles.pop(tile)
            self.knowledge.pop(tile)
            self.take()

        else:

            try:
                self.game.turn_reward -= 1
            except:
                pass

            self.game.lives -= 1
            self.game.board.graveyard.append(t)
            self.tiles.pop(tile)
            self.knowledge.pop(tile)
            self.take()

    def discard(self, tile):

        try:
            self.game.turn_reward -= 0
        except:
            pass

        if tile >= len(self.tiles):
          tile = len(self.tiles) - 1
        self.game.board.graveyard.append(self.tiles[tile])
        self.tiles.pop(tile)
        self.knowledge.pop(tile)
        self.take()
        if self.game.tips < 8:
            self.game.tips += 1

    def tip(self, player, color=None, number=None): #Asegurar que no se tipee a si mismo
        if self.game.tips <= 0:
            return 0
        self.game.tips -= 1
        if color:
            for i, tile in enumerate(player.tiles):
                if tile.color == color:
                    player.knowledge[i].color = color
        else:
            for i, tile in enumerate(player.tiles):
                if tile.number == number:
                    player.knowledge[i].number = number

    def encode_tiles(self):
        tmp = []
        for i in self.tiles:
            tmp += i.encode()
        out = [0]*2*self.game.n_tiles
        out[:len(tmp)] = tmp
        return out

    def encode_knowledge(self):
        tmp = []
        for i in self.knowledge:
            tmp += i.encode()
        out = [0]*2*self.game.n_tiles
        out[:len(tmp)] = tmp
        return out
    
    
class Board():

    def __init__(self):

        self.deck = []
        self.graveyard = []
        self.table = {}

        for color in COLORS:
            for number, amount in enumerate([3, 2, 2, 2, 1]):
                for _ in range(amount):
                    self.deck.append(Tile(color, number+1))

        random.shuffle(self.deck)

        for color in COLORS:
            self.table[color] = 0

    def encode(self):
        pass
    
    
class Game():

    def __init__(self, n_players = 5, easy_mode = False):
        
        self.easy_mode = easy_mode
        
        self.n_players = n_players 
        self.n_tiles = 4 if n_players == 5 else 5
        self.n_actions = 2*self.n_tiles + 10*(n_players - 1)
        self.board = Board()
        self.tips = 8
        self.lives = 3
        self.players = [Player(self) for _ in range(n_players)]
        self.turn = 0
        self.rulebreak = False

        for i in self.players:
            for _ in range(self.n_tiles):
                i.take()


    def __str__(self):

        out = ''
        out += f'Turn: {self.turn}\n'
        out += f'Tips left: {self.tips}\n'
        out += f'Lives left: {self.lives}\n'

        out += 'Tiles in table: '
        out += str(self.board.table)
        out += '\n'

        for i in range(self.n_players):
            out += f'Player{i} tiles: '
            out += str(self.players[i].tiles)
            out += '\n'
            out += f'Player{i} knowledge: '
            out += str(self.players[i].knowledge)
            out += '\n'

        out += f'Deck tiles: '
        out += str(self.board.deck)
        out += '\n'

        out += f'Discarded tiles: '
        out += str(self.board.graveyard)
        out += '\n'

        return out


    def play(self):

        while self.check_finish() == 'play':
            clear_output() #wait=False
            print(self)
            to_play = self.turn % self.n_players
            print('Player to play:', to_play)
            acted = False
            while not acted:
                action = str(input('Action (place, discard, tip): '))
                if action == 'place':
                    tile = int(input('Tile: '))
                    self.players[to_play].place(tile)
                    acted = True
                elif action == 'discard':
                    tile = int(input('Tile: '))
                    self.players[to_play].discard(tile)
                    acted = True
                elif action == 'tip':
                    player = int(input('Player: '))
                    tip = input('Color or number: ')
                    if tip in COLORS:
                        self.players[to_play].tip(self.players[player], color=tip)
                        acted = True
                    else:
                        try:
                            self.players[to_play].tip(self.players[player], number=int(tip))
                            acted = True
                        except:
                            pass
            self.turn += 1

        if self.check_finish() == 'win':
            print('Congratulations!')
        if self.check_finish() == 'loss':
            print(f'Sorry, only {sum(list(self.board.table.values()))} points')


    def play_AI(self, action):

        to_play = self.turn % self.n_players
        self.turn_reward = 0
        #de momento
        action_n = action.tolist()[0][0]

        if action_n < self.n_tiles: #place
            self.players[to_play].place(action_n)
            
        elif action_n < 2*self.n_tiles:  #add is_winnable func for reward?
            n = action_n - self.n_tiles
            self.players[to_play].discard(n)
            
        else:
            n = action_n - 2*self.n_tiles
            target = self.players[(to_play + 1 + n//10) % self.n_players]

            if self.tips <= 0: # No quedan tips
                self.rulebreak = True

            n %= 10 # Different tips for 1 player
            if n < 5:
                self.players[to_play].tip(target, number=n)
            else:
                self.players[to_play].tip(target, color=COLORS[n-5])

        self.turn += 1

        observation = self.encode()

        if self.check_finish() == 'win':
            self.turn_reward += 100
        elif self.check_finish() == 'loss':
            self.turn_reward += sum(self.board.table.values())

        terminated = self.check_finish() != 'play'

        return observation, self.turn_reward, terminated


    def encode(self):

        to_play = self.turn % self.n_players

        out = []
        out.append(self.tips)
        out.append(self.lives)
        out += list(self.board.table.values()) # tiles in table
        
        if self.easy_mode:
            
            out += self.players[to_play].encode_tiles()

            for i in range(self.n_players-1):
                out += self.players[(to_play+i+1) % self.n_players].encode_tiles()
              
                
        else:
            out += self.players[to_play].encode_knowledge()
            
            for i in range(self.n_players-1):
                out += self.players[(to_play+i+1) % self.n_players].encode_tiles()
                out += self.players[(to_play+i+1) % self.n_players].encode_knowledge()

        return out


    def check_finish(self):

        if list(self.board.table.values()) == [5]*5:
            return 'win'
        elif self.lives <= 0:
            return 'loss'
        elif sum([len(i.tiles) for i in self.players]) < self.n_players*(self.n_tiles - 1):
            return 'loss'
        elif self.rulebreak:
            return 'loss'
        else:
            return 'play'
