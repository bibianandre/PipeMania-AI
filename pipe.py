# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 109:
# 94158 Bibiana André
# 106046 Filipe Abreu

from sys import stdin
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

pipe_code = {
    "FE": (0,1,1,1), "FC": (1,0,1,1), "FD": (1,1,0,1), "FB": (1,1,1,0),
    "BE": (0,0,1,0), "BC": (0,0,0,1), "BD": (1,0,0,0), "BB": (0,1,0,0),
    "VE": (0,1,1,0), "VC": (0,0,1,1), "VD": (1,0,0,1), "VB": (1,1,0,0),
    "LH": (0,1,0,1), "LV": (1,0,1,0),
}

class CSP:
    def __init__(self, board):
        self.board = board
        self.variables = self.generate_variables()
        self.domains = self.generate_domains()
        self.constraints = self.generate_constraints()

    def generate_variables(self):
        variables = []
        for row in range(self.board.size):
            for col in range(self.board.size):
                variables.append((row, col))
        return variables

    def generate_domains(self):
        domains = {}
        for variable in self.variables:
            row, col = variable
            pipe = self.board.get_value(row, col)
            domains[variable] = self.board.get_orientations_for_pipe(row, col, pipe)
        return domains

    def generate_constraints(self):
        constraints = []
        for row in range(self.board.size):
            for col in range(self.board.size):
                neighbors = self.board.get_adjacent_cells(row, col)
                for neighbor in neighbors:
                    constraints.append(((row, col), neighbor))
        return constraints

class CSPSolver:
    def __init__(self, csp: CSP):
        self.csp = csp

    def is_consistent(self, variable, value: str, assignment):
        row, col = variable
        for adjacent in self.csp.board.get_adjacent_cells(row, col):
            if adjacent in assignment:
                adjacent_value = assignment[adjacent]
                if not self.pipes_match(value, adjacent_value, row, col, adjacent):
                    return False
        return True

    def pipes_match(self, pipe1: str, pipe2: str, row1: int, col1: int, adjacent):
        row2, col2 = adjacent
        direction = (row2 - row1, col2 - col1)
        pipe1_outlets = pipe_code[pipe1]
        pipe2_outlets = pipe_code[pipe2]
        
        if direction == (-1, 0):  # Up
            return pipe1_outlets[1] == pipe2_outlets[3]
        elif direction == (1, 0):  # Down
            return pipe1_outlets[3] == pipe2_outlets[1]
        elif direction == (0, -1):  # Left
            return pipe1_outlets[0] == pipe2_outlets[2]
        elif direction == (0, 1):  # Right
            return pipe1_outlets[2] == pipe2_outlets[0]
        return False

    def backtrack(self, assignment):
        stack = [(assignment, [])]

        while stack:
            assignment, assigned = stack.pop()
            if len(assignment) == len(self.csp.variables):
                if self.dfs_check(assignment):
                    return assignment
                continue

            unassigned = [v for v in self.csp.variables if v not in assignment]
            first = unassigned[0]

            for value in self.csp.domains[first]:
                if self.is_consistent(first, value, assignment):
                    local_assignment = assignment.copy()
                    local_assignment[first] = value
                    stack.append((local_assignment, assigned + [first]))

        return None

    def dfs_check(self, assignment):
        visited = set()

        stack = [(0, 0)]

        while stack:
            row, col = stack.pop()
            if (row, col) not in visited:
                visited.add((row, col))
                for neighbor in self.adj_cells(row, col, assignment):
                    if neighbor in assignment:  # Check if the cell is assigned
                        stack.append(neighbor)
        return len(visited) == len(self.csp.variables)

    def adj_cells(self, row, col, assignment):
        pipe = pipe_code.get(assignment[(row, col)])
        adjacents = [(row, col - 1), (row - 1, col), (row, col + 1), (row + 1, col)]
        neighbor = []
        for i in range(4):
            if pipe[i] != 0:
                continue
            row_adj, col_adj = adjacents[i]
            if row_adj < 0 or col_adj < 0 or row_adj > self.csp.board.size - 1 or col_adj > self.csp.board.size - 1:
                continue
            neighbor.append(adjacents[i])
        return neighbor
    
    def solve(self):
        """Solve the CSP problem."""
        return self.backtrack({})

class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

class Board:
    """Representação interna de um tabuleiro de PipeMania."""
    def __init__(self, cells):
        """The board consists of cells with pipes of a certain type."""
        self.cells = cells
        self.size = len(cells)
        self.possibilities_per_cell = self.precompute_orientations()
        
    def precompute_orientations(self):
        orientations = {}
        for row in range(self.size):
            for col in range(self.size):
                pipe = self.get_value(row, col)
                orientations[(row, col)] = self.get_orientations_for_pipe(row, col, pipe)
        return orientations

    def get_adjacent_cells(self, row: int, col: int):
        """Return a list of adjacent cells (up, down, left, right) for a given cell."""
        adjacent = []
        if row > 0:  # Up
            adjacent.append((row - 1, col))
        if row < self.size - 1:  # Down
            adjacent.append((row + 1, col))
        if col > 0:  # Left
            adjacent.append((row, col - 1))
        if col < self.size - 1:  # Right
            adjacent.append((row, col + 1))
        return adjacent

    def get_orientations_for_pipe(self, row: int, col: int, pipe: str):
        """Gets valid alternative orientations for a pipe according to its position on the board,
        ensuring that outlets are not directed to the outside of the board.
        """
        constraints = [0, 0, 0, 0]  # Initial constraints [top, right, bottom, left]

        if row == 0:                    # Top Row
            constraints[1] = 1

            if col == 0:                # Top Left Corner
                constraints[0] = 1
            elif col == self.size - 1:  # Top Right Corner
                constraints[2] = 1

        elif row == self.size - 1:      # Bottom Row
            constraints[3] = 1

            if col == 0:                # Bottom Left Corner
                constraints[0] = 1
            elif col == self.size - 1:  # Bottom Right Corner
                constraints[2] = 1

        else:
            if col == 0:                # Left Edge
                constraints[0] = 1
            elif col == self.size - 1:  # Right Edge
                constraints[2] = 1
            else:                       # The Rest
                return [key for key in pipe_code if key.startswith(pipe[0])]

        orientations = [key for key, value in pipe_code.items() if key.startswith(pipe[0]) and
                all((constraints[i] == value[i] or (constraints[i] == 0)) for i in range(4))]
        return orientations
    
    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.cells[row][col].upper()

    def adjacent_vertical_values(self, row: int, col: int):
        """Devolve os valores imediatamente acima e abaixo, respectivamente."""
        return (
            self.get_value(row - 1, col), 
            self.get_value(row + 1, col),
            )

    def adjacent_horizontal_values(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita, respectivamente."""
        return (
            self.get_value(row, col - 1), 
            self.get_value(row, col + 1),
            )

    def clone(self):
        """Creates a copy of the current board """
        cloned_board = Board([row[:] for row in self.cells])
        cloned_board.possibilities_per_cell = {}
        for key, value in self.possibilities_per_cell.items():
            cloned_board.possibilities_per_cell[key] = value
        return cloned_board

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        cells = [line.strip().split() for line in stdin]
        board = Board(cells)
        return board

    def print_board(self):
        for i in range(self.size):
            line = ""
            for j in range(self.size-1):
                line += self.cells[i][j]
                line += '\t'
            line += self.cells[i][j+1]
            print(line)

class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        state = PipeManiaState(board)
        super().__init__(state)
        self.visited = set()

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a partir do estado passado como argumento."""
        actions = []
        for row in range(state.board.size):
            for col in range(state.board.size):
                current_pipe = state.board.get_value(row, col)
                orientations = state.board.possibilities_per_cell[(row, col)]
                for orientation in orientations:
                    if orientation != current_pipe:
                        actions.append((row, col, orientation))
        return actions

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de self.actions(state)."""
        row, col, pipe = action
        new_board = state.board.clone()
        new_board.cells[row][col] = pipe
        return PipeManiaState(new_board)

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        
        if len(self.visited) == state.board.size ** 2:
            return True
        return False
    
    def dfs(self, state, row, col):
            """Depth-First Search implementation."""
            if (row, col) in self.visited:
                return
            self.visited.add((row, col))
            current_pipe = state.board.get_value(row, col)
            current_pipe_value = pipe_code[current_pipe]
            for dr, dc, check1, check2 in [(0, 1, 2, 0), (1, 0, 3, 1), (0, -1, 0, 2), (-1, 0, 1, 3)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < state.board.size and 0 <= new_col < state.board.size:
                    next_pipe = state.board.get_value(new_row, new_col)
                    next_pipe_value = pipe_code[next_pipe]
                    if current_pipe_value[check1] == next_pipe_value[check2]:
                        self.dfs(state, new_row, new_col)
    
    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        return 1


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    board = Board.parse_instance()
    pipemania = PipeMania(board)
    csp = CSP(board)
    solver = CSPSolver(csp)
    solution = solver.solve()

    if solution:
        for (row, col), pipe in solution.items():
            board.cells[row][col] = pipe
        board.print_board()
    else:
        print("No solution found.")
