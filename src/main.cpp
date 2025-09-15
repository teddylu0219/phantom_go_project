#include "minizero/go/go.h"
#include "minizero/utils/sgf_loader.h"
#include "minizero/utils/tqdm.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <set>
#include <numeric>
#include <unordered_set>
#include <deque>
#include <iomanip>

using namespace minizero::utils;
using namespace minizero::env::go;
using minizero::env::Player;

struct MoveInfo {
    bool legal;
    std::vector<int> captured_stones;
};

struct MoveEvent {
    Player mover;
    int pos;
    std::vector<int> captured_stones;
};

struct SeqState {
    std::vector<GoAction> seq;
    GoHashKey hash;
};

struct InfoSetsAtMove {
    std::vector<SeqState> black;
    std::vector<SeqState> white;
};

struct InfoSetSizesRow {
    int move_index;
    size_t black_size;
    size_t white_size;
};

// Create GoEnv from a GoAction sequence
bool rebuildEnvFromSeq(int board_size, const std::vector<GoAction>& seq, GoEnv& env) {
    env = GoEnv(board_size);
    for (const auto& a : seq) {
        if (!env.act(a)) return false;
    }
    return true;
}

bool sameStones(const GoEnv& a, const GoEnv& b) {
    return a.getHashKey() == b.getHashKey();
}

// Analyze a single move's effects
MoveInfo analyzeMove(const GoEnv& before, const GoEnv& after, const GoAction& action) {
    MoveInfo info;
    info.legal = true;

    Player opponent = (action.getPlayer() == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1;
    const auto& old_stones = before.getStoneBitboard();
    const auto& new_stones = after.getStoneBitboard();

    GoBitboard removed = old_stones.get(opponent) & ~new_stones.get(opponent);
    while (!removed.none()) {
        int pos = removed._Find_first();
        removed.reset(pos);
        info.captured_stones.push_back(pos);
    }
    return info;
}

// Build MoveEvent from referee history
std::vector<MoveEvent> buildMoveEvents(
    const std::vector<GoEnv>& referee_history,
    const std::vector<GoAction>& referee_actions)
{
    std::vector<MoveEvent> events;
    events.reserve(referee_actions.size());

    for (size_t k = 0; k < referee_actions.size(); ++k) {
        const GoEnv& before = referee_history[k];
        GoEnv after = before;
        const GoAction& a = referee_actions[k];
        bool ok = after.act(a);
        if (!ok) {
            std::cerr << "CRITICAL: real action illegal at move " << (k+1) << "\n";
            continue;
        }
        MoveInfo info = analyzeMove(before, after, a);
        MoveEvent ev{ a.getPlayer(), a.getActionID(), std::move(info.captured_stones) };
        events.push_back(std::move(ev));
    }
    return events;
}

int countStonesOnBoard(const GoEnv& env, Player p) {
    const int N = env.getBoardSize();
    int cnt = 0;
    for (int i = 0; i < N * N; ++i) {
        if (env.getGrid(i).getPlayer() == p) ++cnt;
    }
    return cnt;
}

std::pair<std::unordered_set<int>, std::unordered_set<int>> computeMustSets(
    int move_number,
    Player perspective, // who knows their own stones
    const std::vector<MoveEvent>& events,
    const std::vector<GoEnv>& referee_history)
{
    const int N = referee_history[0].getBoardSize();
    const int PASS = N * N;
    auto other = [](Player p){ return (p == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1; };

    const Player MY  = perspective;
    const Player OPP = other(MY);

    std::unordered_set<int> must_my, must_opp;

    for (int i = 0; i < move_number; ++i) {
        const auto& ev = events[i];
        const GoEnv& before = referee_history[i];
        const GoEnv& after  = referee_history[i + 1];

        if (ev.mover == MY) {
            // My move => Add it into my MUST
            if (ev.pos != PASS)  {
                must_my.insert(ev.pos);
            }
            // Capture opponent stones => remove opp's MUST
            for (int c : ev.captured_stones) {
                must_opp.erase(c);
            }
        } else { // OPP
            // opponent captureed my stones
            if (!ev.captured_stones.empty()) {
                // this move
                if (ev.pos != PASS) {
                    must_opp.insert(ev.pos);
                }
                // neighbors of captured stones
                for (int s : ev.captured_stones) {
                    for (int nb : before.getGrid(s).getNeighbors()) {
                        if (before.getGrid(nb).getPlayer() == OPP) must_opp.insert(nb);
                    }
                }
            }
            // remove captured stones
            for (int c : ev.captured_stones) {
                must_my.erase(c);
            }
        }
    }

    // Mapping
    std::unordered_set<int> must_black, must_white;
    if (MY == Player::kPlayer1) {
        must_black = std::move(must_my);
        must_white = std::move(must_opp);
    } else {
        must_white = std::move(must_my);
        must_black = std::move(must_opp);
    }
    return {std::move(must_black), std::move(must_white)};
}


// Do not allow capturing already-satisfied MUST stones
bool breaksSatisfiedMust(
    const GoEnv& before, const GoEnv& after, const GoAction& a,
    const std::unordered_set<int>& satisfied_black,
    const std::unordered_set<int>& satisfied_white)
{
    MoveInfo info = analyzeMove(before, after, a);
    for (int c : info.captured_stones) {
        if (a.getPlayer() == Player::kPlayer1) {
            if (satisfied_white.count(c)) return true; // black move captured white MUST
        } else {
            if (satisfied_black.count(c)) return true; // white move captured black MUST
        }
    }
    return false;
}

std::vector<SeqState> sampleInfoSetAtMove(
    int board_size,
    const std::unordered_set<int>& must_black,
    const std::unordered_set<int>& must_white,
    size_t max_info_set_size,
    Player my_perspective,                 // whose perspective we're sampling
    int target_black_count,                // final # of black stones
    int target_white_count,                // final # of white stones
    int max_total_attempts = 20000)
{
    const int PASS = board_size * board_size;
    std::vector<SeqState> out;
    out.reserve(max_info_set_size);
    std::unordered_set<GoHashKey> seen;

    auto other = [](Player p){ return (p == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1; };
    Player opp = other(my_perspective);

    std::mt19937 rng{std::random_device{}()};

    // place one specific stone safely (no breaking MUST)
    auto try_place_specific = [&](GoEnv& env,
                                  int pos,
                                  Player p,
                                  std::unordered_set<int>& satisfied_black,
                                  std::unordered_set<int>& satisfied_white,
                                  std::vector<GoAction>& seq)->bool 
    {
        GoEnv test = env;
        GoAction a(pos, p);
        if (!test.act(a)) return false;
        if (breaksSatisfiedMust(env, test, a, satisfied_black, satisfied_white)) return false;
        env = std::move(test);
        seq.push_back(a);
        if (p == Player::kPlayer1) {
            satisfied_black.insert(pos);
        } else {
            satisfied_white.insert(pos);
        }                      
        return true;
    };

    std::vector<int> steps(max_total_attempts);
    std::iota(steps.begin(), steps.end(), 1);
    auto pbar = minizero::utils::tqdm::tqdm(
        steps, "Sampling attempts: {step}/{size}");

    for (int attempt : pbar) {
        if (out.size() >= max_info_set_size) break;

        GoEnv env(board_size);
        std::vector<GoAction> seq;
        seq.reserve(128);

        // MUST stones
        std::unordered_set<int> pending_black = must_black;
        std::unordered_set<int> pending_white = must_white;
        std::unordered_set<int> satisfied_black, satisfied_white;

        bool fail = false;
        // Initially black turn
        Player turn = Player::kPlayer1;

        // Step 1: place all MUST stones
        while ((!pending_black.empty() || !pending_white.empty()) && !fail) {
            auto& pending_me   = (turn == Player::kPlayer1) ? pending_black : pending_white;
            bool placed = false;

            if (!pending_me.empty()) {
                std::vector<int> cand(pending_me.begin(), pending_me.end());
                std::shuffle(cand.begin(), cand.end(), rng);
                for (int pos : cand) {
                    if (try_place_specific(env, pos, turn, satisfied_black, satisfied_white, seq)) {
                        pending_me.erase(pos);
                        placed = true;
                        break;
                    }
                }
            }

            if (!placed) {
                // MUAT is empty => PASS
                GoEnv test = env;
                GoAction pass(PASS, turn);
                if (!test.act(pass)) { fail = true; break; }
                env = std::move(test);
                seq.push_back(pass);
            }

            turn = other(turn);
        }
        if (fail) continue;

        // all MUST are already placed 
        if (!pending_black.empty() || !pending_white.empty()) continue;

        // Randomly place the opponent's stones until the target number reaches
        int cur_black = countStonesOnBoard(env, Player::kPlayer1);
        int cur_white = countStonesOnBoard(env, Player::kPlayer2);

        int need_black = std::max(0, target_black_count - cur_black);
        int need_white = std::max(0, target_white_count - cur_white);

        int need_opp   = (opp == Player::kPlayer1) ? need_black : need_white;
        int need_self  = 0; 

        int safety_steps = board_size * board_size * 4; // 防暴衝
        while (!fail && safety_steps-- > 0 && (need_opp > 0 || need_self > 0)) {
            if (turn == my_perspective) {
                // PASS for my turn
                GoEnv test = env;
                GoAction pass(PASS, turn);
                if (!test.act(pass)) { fail = true; break; }
                env = std::move(test);
                seq.push_back(pass);
            } else {
                // opp's turn => randomly place (do not break MUST)
                std::vector<GoAction> legal = env.getLegalActions();
                if (legal.empty()) { fail = true; break; }
                std::shuffle(legal.begin(), legal.end(), rng);

                bool moved = false;
                for (const auto& a : legal) {
                    int id = a.getActionID();
                    if (id == PASS) continue;
                    if (a.getPlayer() != opp) continue;

                    GoEnv test = env;
                    if (!test.act(a)) continue;

                    if (breaksSatisfiedMust(env, test, a, satisfied_black, satisfied_white)) continue;

                    env = std::move(test);
                    seq.push_back(a);
                    moved = true;
                    if (opp == Player::kPlayer1) {
                        --need_black, --need_opp;
                    } else {
                        --need_white, --need_opp;
                    }              
                    break;
                }
                if (!moved) { fail = true; break; }
            }
            turn = other(turn);
        }

        if (fail) continue;

        // check if reaching the target number of stones
        int fin_black = countStonesOnBoard(env, Player::kPlayer1);
        int fin_white = countStonesOnBoard(env, Player::kPlayer2);
        if (fin_black != target_black_count || fin_white != target_white_count) continue;

        // hash
        GoHashKey h = env.getHashKey();
        if (seen.insert(h).second) {
            out.push_back(SeqState{ std::move(seq), h });
        }
    }

    return out;
}


// Ensure ground truth exists in the info set
void ensureGroundTruthIncluded(
    int move_number,
    const std::vector<GoAction>& referee_actions,
    int board_size,
    size_t max_info_set_size,
    std::vector<SeqState>& seqs)
{
    std::vector<GoAction> truth_seq(referee_actions.begin(),
                                    referee_actions.begin() + move_number);

    GoEnv env(board_size);
    if (!rebuildEnvFromSeq(board_size, truth_seq, env)) {
        std::cerr << "ensureGroundTruthIncluded: rebuild failed for truth_seq\n";
        return;
    }
    GoHashKey truth_hash = env.getHashKey();

    // check if already exists
    bool present = false;
    for (const auto& s : seqs) {
        if (s.hash == truth_hash) { 
            present = true;
            break; 
        }
    }
    if (present) return;

    // if not, add it into the information set
    SeqState gt{ std::move(truth_seq), truth_hash };
    if (seqs.size() < max_info_set_size) {
        seqs.push_back(std::move(gt));
    } else {
        static std::mt19937 rng{std::random_device{}()};
        std::uniform_int_distribution<size_t> dist(0, seqs.size() - 1);
        seqs[dist(rng)] = std::move(gt);
    }
}

InfoSetsAtMove findInfoSetsAtMove(
    int move_number,
    const std::vector<GoEnv>& referee_history,
    const std::vector<GoAction>& referee_actions,
    size_t max_info_set_size,
    InfoSetSizesRow* size_row = nullptr)
{
    InfoSetsAtMove ret;
    if (move_number < 0 || move_number > (int)referee_actions.size()) {
        std::cerr << "Error: Invalid move_number " << move_number << "\n";
        return ret;
    }
    const int N = referee_history[0].getBoardSize();

    auto events = buildMoveEvents(referee_history, referee_actions);

    // Target number of stones
    const GoEnv& truth = referee_history[move_number];
    auto countStones = [&](Player p){
        int c = 0;
        for (int i = 0; i < N*N; ++i) {
            if (truth.getGrid(i).getPlayer() == p) {
                ++c;
            }
        }
        return c;
    };
    const int target_black = countStones(Player::kPlayer1);
    const int target_white = countStones(Player::kPlayer2);

    {
        auto [mb, mw] = computeMustSets(move_number, Player::kPlayer1, events, referee_history);
        std::cout << "\n[Sampling attempts: Black]\n";
        ret.black = sampleInfoSetAtMove(
            N, move_number, mb, mw, max_info_set_size,
            Player::kPlayer1,
            target_black, target_white);
        ensureGroundTruthIncluded(move_number, referee_actions, N, max_info_set_size, ret.black);
    }

    {
        auto [mb, mw] = computeMustSets(move_number, Player::kPlayer2, events, referee_history);
        std::cout << "\n[Sampling attempts: White]\n";
        ret.white = sampleInfoSetAtMove(
            N, move_number, mb, mw, max_info_set_size,
            Player::kPlayer2,
            target_black, target_white);
        ensureGroundTruthIncluded(move_number, referee_actions, N, max_info_set_size, ret.white);
    }

    if (size_row) {
        size_row->move_index = move_number;
        size_row->black_size = ret.black.size();
        size_row->white_size = ret.white.size();
    }

    return ret;
}


// Display and verification
void analyzeAndDisplayInfoSet(
    Player player,
    int move_number,
    const std::vector<GoEnv>& referee_history,
    const std::vector<GoAction>& referee_actions,
    size_t max_info_set_size)
{
    if (move_number <= 0 || move_number > (int)referee_actions.size()) {
        std::cerr << "Error: Invalid move number " << move_number << "\n";
        return;
    }
    const int board_size = referee_history[0].getBoardSize();
    std::string player_name = (player == Player::kPlayer1) ? "Black" : "White";

    std::cout << "\n=================================================================\n"
              << "Calculating Information Set for " << player_name
              << " after move #" << move_number << "...\n"
              << "=================================================================\n\n";

    // Generate the information set at move
    
    InfoSetSizesRow size_row{};
    auto both = findInfoSetsAtMove(
        move_number, referee_history, referee_actions, max_info_set_size, &size_row);
    const auto& seq_set = (player == Player::kPlayer1) ? both.black : both.white;

    // print the real board
    const GoEnv& truth = referee_history[move_number];
    std::cout << "\n[Real referee board after move #" << move_number << "]\n" << truth.toString() << "\n";
    
    std::cout << "[InfoSet perspective: " << player_name << "]\n";

    // pick 5 boards from the information set randomly
    {
        static std::mt19937 rng{std::random_device{}()};
        std::vector<size_t> idx(seq_set.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        size_t show = std::min<size_t>(5, idx.size());
        for (size_t i = 0; i < show; ++i) {
            GoEnv env(board_size);
            rebuildEnvFromSeq(board_size, seq_set[idx[i]].seq, env);
            std::cout << "Possibility #" << (i + 1) << ":\n" << env.toString() << "\n";
        }
    }


    std::cout << "\nInformation Set Size\n";
    const int W1 = 6, W2 = 12, W3 = 12;
    std::cout << std::left  << std::setw(W1) << "Move"
            << std::right << std::setw(W2) << "Black"
            << std::setw(W3) << "White" << '\n';

    std::cout << std::left  << std::setw(W1) << size_row.move_index
            << std::right << std::setw(W2) << size_row.black_size
            << std::setw(W3) << size_row.white_size << '\n';
}

// load SGF, build referee history, run demo
int main() {
    SGFLoader loader;
    if (!loader.loadFromFile("../test.sgf")) {
        std::cerr << "Error: Cannot load .sgf file" << std::endl;
        return 1;
    }

    std::cout << "Sucessfully load .sgf file; totally " << loader.getActions().size() << " moves" << std::endl;

    int board_size = 9;
    minizero::config::env_board_size = board_size;
    initialize();
    const size_t MAX_INFO_SET_SIZE = 10000;

    std::vector<GoEnv> referee_history;
    std::vector<GoAction> referee_actions;

    GoEnv current_env(board_size);
    referee_history.push_back(current_env);

    for (const auto& action_pair : loader.getActions()) {
        const auto& sgf_action = action_pair.first;
        if (sgf_action.size() < 2) continue;

        Player player = (sgf_action[0] == "B") ? Player::kPlayer1 : Player::kPlayer2;
        int action_id;
        if (sgf_action[1].empty() || sgf_action[1] == "tt" || sgf_action[1] == "pass") {
            action_id = board_size * board_size; // PASS
        } else {
            action_id = SGFLoader::boardCoordinateStringToActionID(sgf_action[1], board_size);
        }

        GoAction action(action_id, player);
        if (!current_env.act(action)) {
            std::cerr << "Invalid move: " << referee_actions.size() + 1 << std::endl;
            return 1;
        }

        referee_actions.push_back(action);
        referee_history.push_back(current_env);
    }

    int target_move = 40;
    analyzeAndDisplayInfoSet(Player::kPlayer2, target_move, referee_history, referee_actions, MAX_INFO_SET_SIZE);

    return 0;
}
