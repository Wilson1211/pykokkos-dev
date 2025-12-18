import pykokkos as pk


@pk.workunit
def init_data(i, view):
    view[i] = i + 1


# Test lower_bound with scratch memory
@pk.workunit
def team_lower_bound(team_member, view, result_view):
    team_size: int = team_member.team_size()
    offset: int = team_member.league_rank() * team_size
    localIdx: int = team_member.team_rank()
    globalIdx: int = offset + localIdx
    team_rank: int = team_member.team_rank()

    scratch: pk.ScratchView1D[int] = pk.ScratchView1D(
        team_member.team_scratch(0), team_size
    )

    scratch[team_rank] = view[globalIdx]
    team_member.team_barrier()
    search_value: int = team_rank * 2  # Search for a value
    bound_idx: int = pk.lower_bound(scratch, team_size, search_value)
    result_view[globalIdx] = bound_idx


# Test lower_bound with regular view
# Find lower bound for value i in the first 10 elements
@pk.workunit
def lower_bound_view(i, view, result_view):
    search_value: int = i
    bound_idx: int = pk.lower_bound(view, 10, search_value)
    result_view[i] = bound_idx


def main():
    N = 64
    team_size = 32
    num_teams = (N + team_size - 1) // team_size

    view: pk.View1D[int] = pk.View([N], int)
    result_view: pk.View1D[int] = pk.View([N], int)

    # Expected results
    expected_scratch = pk.View([64], int)
    expected_scratch_data = [
        0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29
    ]
    for i in range(64):
        expected_scratch[i] = expected_scratch_data[i]

    expected_view = pk.View([64], int)
    expected_view_data = [
        0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10
    ]
    for i in range(64):
        expected_view[i] = expected_view_data[i]

    p_init = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, N)
    pk.parallel_for(p_init, init_data, view=view)

    print(f"Total elements: {N}, Team size: {team_size}, Number of teams: {num_teams}")
    print(f"Initial view: {view}")

    # Test with TeamPolicy (scratch memory)
    team_policy = pk.TeamPolicy(pk.ExecutionSpace.OpenMP, num_teams, team_size)

    pk.parallel_for(team_policy, team_lower_bound, view=view, result_view=result_view)
    print(f"Result (scratch lower_bound): {result_view}")

    # Assert scratch lower_bound results
    for i in range(N):
        assert (
            result_view[i] == expected_scratch[i]
        ), f"Mismatch at index {i}: got {result_view[i]}, expected {expected_scratch[i]}"

    # Test with RangePolicy (regular view)
    pk.parallel_for(p_init, lower_bound_view, view=view, result_view=result_view)
    print(f"Result (view lower_bound): {result_view}")

    for i in range(N):
        assert (
            result_view[i] == expected_view[i]
        ), f"Mismatch at index {i}: got {result_view[i]}, expected {expected_view[i]}"


if __name__ == "__main__":
    main()
