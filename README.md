## Compilation & Execution
```shell
mkdir build
cd build
cmake ..
make
./main
```
The program loads a game from test.sgf and generates information sets for a specified move.

## Information set generator algorithm

### 1. Compute MUST Sets

- MUST stones: Stones that a player can definitively deduce exist on the board
  
- For each player's perspective:
  - Own moves are always known (added to own MUST set)
  - Opponent moves revealed through captures:
    - The capturing stone itself
    - Opponent stones adjacent to own captured stones
  - Captured stones are removed from MUST sets

### 2. Sample Board States

- Phase 1: Place all MUST stones
  - Alternating turns, place MUST stones for each player
  - Ensure no illegal moves or breaking of already-satisfied MUST constraints
  - Pass if no valid MUST stone placement available

- Phase 2: Random placement of opponent's unknown stones
  - Add opponent's stones randomly until reaching the correct stone count
  - Own player passes during this phase
  - Avoid capturing already-placed MUST stones
