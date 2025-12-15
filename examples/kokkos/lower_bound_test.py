import pykokkos as pk


@pk.workunit
def init_data(i: int, view: pk.View1D[int]):
    view[i] = i + 1


# Test lower_bound with scratch memory
@pk.workunit
def team_lower_bound(team_member: pk.TeamMember, view: pk.View1D[int], result_view: pk.View1D[int]):
    team_size: int = team_member.team_size()
    offset: int = team_member.league_rank() * team_size
    localIdx: int = team_member.team_rank()
    globalIdx: int = offset + localIdx
    team_rank: int = team_member.team_rank()

    # Allocate scratch memory for sorted data
    scratch: pk.ScratchView1D[int] = pk.ScratchView1D(
        team_member.team_scratch(0), team_size
    )

    # Copy data to scratch and make it sorted within the team
    scratch[team_rank] = view[globalIdx]
    team_member.team_barrier()

    # Now use lower_bound to find position in scratch
    # For example, find lower bound for the value at team_rank position
    search_value: int = team_rank * 2  # Search for a value
    
    # Find lower bound in scratch memory
    bound_idx: int = pk.lower_bound(scratch, team_size, search_value)
    
    # Store result
    result_view[globalIdx] = bound_idx


# Test lower_bound with regular view
@pk.workunit
def test_lower_bound_view(i: int, view: pk.View1D[int], result_view: pk.View1D[int]):
    # Find lower bound for value i in the first 10 elements
    search_value: int = i
    bound_idx: int = pk.lower_bound(view, 10, search_value)
    result_view[i] = bound_idx


def main():
    N = 64
    team_size = 32
    num_teams = (N + team_size - 1) // team_size

    view: pk.View1D[int] = pk.View([N], int)
    result_view: pk.View1D[int] = pk.View([N], int)
    
    p_init = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, N)
    pk.parallel_for(p_init, init_data, view=view)

    print(f"Total elements: {N}, Team size: {team_size}, Number of teams: {num_teams}")
    print(f"Initial view: {view}")

    # Test with TeamPolicy (scratch memory)
    team_policy = pk.TeamPolicy(pk.ExecutionSpace.OpenMP, num_teams, team_size)
    
    print("\nRunning lower_bound with scratch memory...")
    pk.parallel_for(team_policy, team_lower_bound, view=view, result_view=result_view)
    print(f"Result (scratch lower_bound): {result_view}")

    # Test with RangePolicy (regular view)
    print("\nRunning lower_bound with regular view...")
    pk.parallel_for(p_init, test_lower_bound_view, view=view, result_view=result_view)
    print(f"Result (view lower_bound): {result_view}")


if __name__ == "__main__":
    main()
