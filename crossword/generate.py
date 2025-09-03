import sys

from crossword import *

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # god I love python hahaha
        self.domains = { v: set(w for w in self.domains[v] if len(w) == v.length) for v in self.domains }

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        overlap = self.crossword.overlaps[x, y]

        if overlap is None:
            return False

        ox, oy = overlap

        to_remove = set()

        for x_word in self.domains[x]:
            possible = False
            for y_word in self.domains[y]:
                curr_possible = True
                if x_word[ox] != y_word[oy]:
                    curr_possible = False
                if curr_possible:
                    possible = True
            if not possible:
                to_remove.add(x_word)

        self.domains[x] = { w for w in self.domains[x] if w not in to_remove }

        return len(to_remove) != 0

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = []
            for x in self.domains:
                for y in self.crossword.neighbors(x):
                    arcs.append((x, y))

        while len(arcs) != 0:
            x, y = arcs.pop(0)
            if self.revise(x, y):
                for z in self.crossword.neighbors(x):
                    arcs.append((z, x))

        for x in self.domains:
            if len(self.domains[x]) == 0:
                return False

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        return self.domains.keys() == assignment.keys()

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        word_set = set()
        # good variable naming right here ^^
        for ass in assignment:
            w = assignment[ass]
            if len(w) != ass.length:
                return False

            word_set.add(w)

        if len(word_set) != len(assignment):
            return False

        for x in assignment:
            for y in self.crossword.neighbors(x):
                if y not in assignment:
                    continue

                overlap = self.crossword.overlaps[x, y]
                if overlap == None:
                    continue

                if assignment[x][overlap[0]] != assignment[y][overlap[1]]:
                    return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        def get_n_of_values_ruled_out(word):
            count = 0
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    continue

                overlap = self.crossword.overlaps[var, neighbor]
                if overlap is None:
                    continue

                ox, oy = overlap
                for possible_other_word in self.domains[neighbor]:
                    if word[ox] != possible_other_word[oy]:
                        count += 1

            return count


        return sorted(self.domains[var], key=lambda w: get_n_of_values_ruled_out(w))
        # return sorted(var, key=)


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        possible_vars = set()
        for var in self.domains:
            if var not in assignment:
                possible_vars.add(var)

        return min(possible_vars, key=lambda x: len(self.domains[x]))

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        x = self.select_unassigned_variable(assignment)

        for w in self.order_domain_values(x, assignment):
            curr = assignment.copy()
            curr[x] = w
            if not self.consistent(curr):
                continue
            res = self.backtrack(curr)
            if res:
                return res

        return None

def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
